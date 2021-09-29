# SsENet.PyTorch
An experimental implementation to verify variation idea to [SENet(Squeeze-and-Excitation Networks)](https://arxiv.org/abs/1709.01507).
'Ss' in title means the variant is applied to Squeeze operation. I'm so sad that I can't write [ß](https://ko.wikipedia.org/wiki/%C3%9F) in repository name... How cool is `ßENet.PyTorch`!

## Prerequisites
This code is under following environment:
- python=3.9.4
- pytorch=1.9.1
- torchvision=0.10.1

## Variant on Squeeze operation
![SENet process](https://user-images.githubusercontent.com/30234176/134904288-81826228-015e-4cb7-873e-fec1f891769d.png)
![global average pooling](https://user-images.githubusercontent.com/30234176/134905250-4797bbd9-011b-442c-a43e-7c47ede033d4.png)

In SENet, to exploit input-specific descriptor **_z_**, **global-average-pooling**(nn.AdaptiveAvgPool2d on PyTorch) is used - it is just the average value of each 'image' whose size is `H * W`.

In my opinion, the **standard-deviation**(torch.std on PyTorch) is more effective value to represent each channel's importance. If one channel has high std, it means this channel has ability to distinguish between background and object.  

So, In this project, I'll verify that **standard-deviation** is efficient to express channel's importance by conducting experiments.
