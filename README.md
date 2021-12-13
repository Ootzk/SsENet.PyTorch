# SsENet.PyTorch
An experimental implementation to verify variation idea to [SENet(Squeeze-and-Excitation Networks)](https://arxiv.org/abs/1709.01507).

## Prerequisites
This code is under following environment:
- python=3.9.4
- pytorch=1.9.1
- torchvision=0.10.1

## Explanation of Original Squeeze operation
![SENet process](https://user-images.githubusercontent.com/30234176/134904288-81826228-015e-4cb7-873e-fec1f891769d.png)  
Above figure shows the concept idea of SENet. They **squeezed** by taking representative value(scalar) per channel. (`H * W * C` -> `1 * 1 * C`)
![global average pooling](https://user-images.githubusercontent.com/30234176/134905250-4797bbd9-011b-442c-a43e-7c47ede033d4.png)  
In original paper, to exploit input-specific descriptor **_z_**, **global-average-pooling**(nn.AdaptiveAvgPool2d on PyTorch) was used - it is just the average value of each 'image' whose size is `H * W`.

## Variant Squeeze operations
We've conducted experiments under same training recipe but only different squeeze operation. The squeeze operation we used are:
1. baseline (no squeeze operation; naive architecture)
2. gap (global average pooling)
3. gmp (global max pooling)
4. std (standard deviation of pixels in `H * W`)
5. gapXstd (gap * std)
6. random (random value that have distribution on the interval `[0, 1)`. **not extracted from `H * W`**)

## Experimental Results
- ResNet18 + ImageNet

| Squeeze operation | Top-1 Acc (%) | Top-5 Acc (%) | Improved Top-1 %p |
| ---------- | ---------- | ----------| ---------- |
| baseline | 65.71 | 86.29 | +0.00 |
| gap | 66.91 | 87.35 | +1.20 |
| gmp | 66.73 | 87.20 | +1.02 |
| std | 66.47 | 86.94 | +0.76 |
| gapXstd | 66.90 | 87.18 | +1.19 |
| random | 65.36 | 86.22 | -0.35 |

- ResNet50 + ImageNet

| Squeeze operation | Top-1 Acc (%) | Top-5 Acc (%) | Improved Top-1 %p |
| ---------- | ---------- | ----------| ---------- |
| baseline | 70.94 | 89.84 | +0.00 |
| gap | 70.12 | 89.34 | -0.82 |
| gmp | 70.36 | 89.23 | -0.58 |
| std | 70.34 | 89.33 | -0.60 |
| gapXstd | 70.17 | 89.26 | -0.77 |
| random | 68.99 | 88.71 | -1.95 |