# DeepLearningExperimentalPlatform
My custom deep learning experimental platform. This template supports automated custom deep learning model training and record the results.

## Prerequisite
I assume that you have at least 3 dataset - CIFAR10, CIFAR100, ImageNet on '/dataset' folder.
You can modify default dataset location at train.py, also you can freely install more datasets.

## Structure of Template
- architectures: In default, Baseline(naive one) / Quantization_Aware_Training(QAT) / Squeeze_and_Excitation(SE) is located. You can delete existing architectures and add your experimental architectures freely.
- materials: sample batches from real dataset(CIFAR and ImageNet) to visualize your architecture's logic, output, ... are located here, also you can put the images/figures here to explain your algorithm more detail.
- Q: Queue to accomodate the series of experiments. In default, sample configurations of Baseline/QAT/SE are located in.
- analyze.py: tools and helpful functions to analyze.
- train.py: train related components and functions, also it is the training script itself.
- unittest.ipynb: visual unittest of your models(based on architectures) and components works well with samples in 'materials' folder.

### After generate
After you generate the repository based on this template, **you have to add 'experiments' folder** which will archieve your experiment results(including models, config files, logs, ...)
Also you have to unstage materials/, unittest.ipynb, cause they are heavy and non-essential components.

## Working Process
1. write your (experiment) architectures at architectures/ folder.
2. check that works well with unittest.ipynb.
3. push the configs to experiment to Q/
4. run the script: 
> nohup python train.py Q > log.out; jobs; tail -f log.out
5. analyze the results with analyze.py. (maybe you can make new notebook to visualize)

## Sample Result
![image](https://user-images.githubusercontent.com/30234176/135199986-029d9fbc-aa20-4776-84ed-7707e7b96eb2.png)
Sample configuration's train result log visualized by tensorboard.
