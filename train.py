"""
train.py

model train related components and functions.
"""
from datetime import datetime
import json
import os
from pathlib import Path
from PIL import ImageFile
import numpy as np
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, Resize, Normalize, ToTensor

import ignite
import ignite.distributed as idist
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine import create_supervised_evaluator, Engine, State, Events, _prepare_batch
from ignite.metrics import Loss, Accuracy, TopKCategoricalAccuracy, RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed

from architectures import *

###############################################################################################################
seed = 123

manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

save_architectures = True

###############################################################################################################
dataset_location = {
    'CIFAR10':  '/dataset/CIFAR10',
    'CIFAR100': '/dataset/CIFAR100',
    'ImageNet': '/dataset/ImageNet/Classification'
}

normalization = {
    'CIFAR10':  Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'CIFAR100': Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'ImageNet': Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
    'DEBUG':    Normalize((0.5000, 0.5000, 0.5000), (0.2000, 0.2000, 0.2000))
}

creators = {
    'resnet18_Baseline': resnet18_Baseline,
    'resnet34_Baseline': resnet34_Baseline,
    'resnet50_Baseline': resnet50_Baseline,
    
    'resnet18_SE': resnet18_SE,
    'resnet34_SE': resnet34_SE,
    'resnet50_SE': resnet50_SE,
    
    'resnet18_SsE': resnet18_SsE,
    'resnet34_SsE': resnet34_SsE,
    'resnet50_SsE': resnet50_SsE,
    
    'resnet18_SssE': resnet18_SssE,
    'resnet34_SssE': resnet34_SssE,
    'resnet50_SssE': resnet50_SssE
}

###############################################################################################################
def get_model_skeleton(model_config, target_dataset):
    variation = model_config['variation']['type']
    if variation is None:
        variation = 'Baseline'
        
    if variation not in ['Baseline', 'SE', 'SsE', 'SssE']:
        raise NotImplementedError(f'variation {variation} does not supported')
        
    arch = f'{model_config["backbone"]}_{variation}'
    return creators[arch](target_dataset, model_config['variation']['config'])



def get_transform(dataset):
    if dataset == 'CIFAR10':
        train_transform = Compose([RandomCrop((32, 32), 4), RandomHorizontalFlip(), ToTensor(), normalization[dataset]])
        eval_transform  = Compose([ToTensor(), normalization[dataset]])
    elif dataset == 'CIFAR100':
        train_transform = Compose([RandomCrop((32, 32), 4), RandomHorizontalFlip(), ToTensor(), normalization[dataset]])
        eval_transform  = Compose([ToTensor(), normalization[dataset]])
    elif dataset == 'ImageNet':
        train_transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor(), normalization[dataset]])
        eval_transform  = Compose([Resize(256), CenterCrop(224), ToTensor(), normalization[dataset]])
    else: # fake dataset
        train_transform = Compose([ToTensor(), normalization[dataset]])
        eval_transform  = Compose([ToTensor(), normalization[dataset]])
    
    return train_transform, eval_transform


def get_train_eval_datasets(dataset):
    train_transform, eval_transform = get_transform(dataset)
    
    if dataset == 'DEBUG':
        train_ds = datasets.FakeData(size=640, transform=train_transform)
        eval_ds  = datasets.FakeData(size=640, transform=eval_transform)
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        train_ds = datasets.__dict__[dataset](root=dataset_location[dataset], train=True, transform=train_transform)
        eval_ds  = datasets.__dict__[dataset](root=dataset_location[dataset], train=False, transform=eval_transform)
    elif dataset == 'ImageNet':
        train_ds = datasets.ImageFolder('{0}/train'.format(dataset_location[dataset]), transform=train_transform)
        eval_ds = datasets.ImageFolder('{0}/val'.format(dataset_location[dataset]), transform=eval_transform)
    
    return train_ds, eval_ds



def get_save_handler(config):
    title = config['experiment']['title']
    ID = str(config['experiment']['ID']).zfill(2)
    
    return DiskSaver('experiments/{0}/{1}/checkpoints'.format(title, ID), require_empty=True, create_dir=True)



def get_tb_logger(config):
    title = config['experiment']['title']
    ID = str(config['experiment']['ID']).zfill(2)
    
    return SummaryWriter('experiments/{0}/{1}/tensorboards'.format(title, ID))



def get_dataloader(config):
    if idist.get_rank() > 0:
        idist.barrier()
    
    train_dataset, eval_dataset = get_train_eval_datasets(config['dataloader']['dataset'])
    
    if idist.get_rank() == 0:
        idist.barrier()
        
    train_dataloader = idist.auto_dataloader(train_dataset, **config['dataloader']['train_loader_params'])
    eval_dataloader = idist.auto_dataloader(eval_dataset, **config['dataloader']['eval_loader_params'])
    
    return {'train': train_dataloader, "eval": eval_dataloader}



def create_trainer(model, optimizer, loss_function, scheduler, config): 
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=idist.device(), non_blocking=True)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        if config['gradient_clipping'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping'])
        optimizer.step()
        
        return {'loss': loss.detach(), 'y_pred': y_pred, 'y': y}
    
    trainer = DeterministicEngine(_update)
    
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'Train/Loss')
    RunningAverage(Accuracy(output_transform=lambda x: [x['y_pred'], x['y']])).attach(trainer, 'Train/Top-1')
    RunningAverage(TopKCategoricalAccuracy(k=5, output_transform=lambda x: [x['y_pred'], x['y']])).attach(trainer, 'Train/Top-5')
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def schedule(engine):
        scheduler.step()
            
    if config['resume_from'] is not None:
        checkpoint_fp = Path(config['resume_from'])
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location='cpu')
        Checkpoint.load_objects(to_load={"trainer": trainer, "model": model, "optimizer": optimizer},
                                checkpoint=checkpoint)
        model = model.to(idist.device())
        scheduler.optimizer = optimizer
        scheduler.last_epoch = trainer.state.epoch
                
    return trainer



def create_evaluator(model, config):
    evaluator = create_supervised_evaluator(model, device=idist.device(), non_blocking=True)
    
    Accuracy(output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'Eval/Top-1')
    TopKCategoricalAccuracy(k=5, output_transform=lambda x: [x[0], x[1]]).attach(evaluator, 'Eval/Top-5')
        
    return evaluator



def register_tb_logger_handlers(trainer, evaluator, optimizer, tb_logger, config):
    @trainer.on(Events.ITERATION_COMPLETED(every=config['train_constants']['log_train_stats_every_iters']))
    def _log_train_statistics(engine):
        tb_logger.add_scalar('Train/Loss', engine.state.metrics['Train/Loss'], global_step=engine.state.iteration)
        tb_logger.add_scalar('Train/Top-1', engine.state.metrics['Train/Top-1'], global_step=engine.state.iteration)
        tb_logger.add_scalar('Train/Top-5', engine.state.metrics['Train/Top-5'], global_step=engine.state.iteration)
        
    @evaluator.on(Events.COMPLETED)
    def _log_evaluation_statistics(engine):
        tb_logger.add_scalar('Eval/Top-1', engine.state.metrics['Eval/Top-1'], global_step=trainer.state.epoch)
        tb_logger.add_scalar('Eval/Top-5', engine.state.metrics['Eval/Top-5'], global_step=trainer.state.epoch) 
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def _log_train_params(engine):
        tb_logger.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step=engine.state.epoch)
        
        
        
def prepare(config):
    model = get_model_skeleton(config['model'], config['dataloader']['dataset'])
    model = model.to(idist.device())
    model = DDP(model, device_ids=[idist.get_local_rank()], find_unused_parameters=True)
    
    optimizer = optim.__dict__[config['optimizer']['type']](
        model.parameters(), **config['optimizer']['params']
    )
    optimizer = idist.auto_optim(optimizer)
    
    loss_function = nn.__dict__[config['loss_function']['type']]().to(idist.device())
    
    scheduler = optim.lr_scheduler.__dict__[config['scheduler']['type']](
        optimizer, **config['scheduler']['params']
    )
    
    return model, optimizer, loss_function, scheduler
        
        
        
def train_process(local_rank, config):
    dataloader = get_dataloader(config)
    model, optimizer, loss_function, scheduler = prepare(config)
    trainer = create_trainer(model, optimizer, loss_function, scheduler, config)
    evaluator = create_evaluator(model, config)    
    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "scheduler": scheduler}
    
    @trainer.on(Events.EPOCH_COMPLETED(every=config['train_constants']['save_every_epochs']) | Events.COMPLETED)
    def evaluate(engine):
        evaluator.run(dataloader['eval'])
        
    checkpoint_handler = Checkpoint(
        to_save=to_save,
        save_handler=get_save_handler(config),
        score_function=lambda engine: engine.state.metrics['Eval/Top-1'],
        score_name='eval_top1_accuracy',
        n_saved=None,
        global_step_transform=global_step_from_engine(trainer)
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)
    
    if idist.get_rank() == 0:
        tb_logger = get_tb_logger(config)
        register_tb_logger_handlers(trainer, evaluator, optimizer, tb_logger, config)
    
    try:
        trainer.run(dataloader['train'], max_epochs=config['train_constants']['max_epochs'])
        evaluator.run(dataloader['eval'])
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        
    if idist.get_rank() == 0:
        tb_logger.close()
        
        
        
def run(config):
    with idist.Parallel(backend='nccl', nproc_per_node=8) as parallel:
        parallel.run(train_process, config)
        
        
        
def save_current_component_and_model_files_and_given_configuration_file(config_as_argument):
    with open(config_as_argument, 'r') as config_file:
        config = json.load(config_file)
        
        title = config['experiment']['title']
        ID = str(config['experiment']['ID']).zfill(2)
        
        os.makedirs('experiments/{0}/{1}'.format(title, ID))
        config['datetime'] = datetime.now().strftime("%Y%m%d")
        with open('experiments/{0}/{1}/config.json'.format(title, ID), 'w') as save_config_file:
            json.dump(config, save_config_file, indent=4)
    
    if save_architectures:
        shutil.copytree('architectures',
                        'experiments/{0}/{1}/architectures'.format(title, ID))
    return config
        
    
        
if __name__ == "__main__":
    if os.path.isdir(sys.argv[1]):
        for filename in sorted(os.listdir(sys.argv[1])):
            if filename.endswith('.json'):
                config = save_current_component_and_model_files_and_given_configuration_file('Q/{0}'.format(filename))
                run(config)
    else:
        config = save_current_component_and_model_files_and_given_configuration_file(sys.argv[1])
        run(config)