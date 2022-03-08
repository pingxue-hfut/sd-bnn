
# SD-BNN

This project is the PyTorch implementation of our paper : Self-Distribution Binary Neural Networks. 

**Binarization:** It is consistent with other BNNs tha we binarize all other convolutional layers except for the first and last layers of the networks.

**Data augmentation:** We use the same operations as the existing BNNs do on the datasets.

**Training:** To train the networks from scratch, we use the EDE method during the back propagation process which is proposed in IR-Net[1] as our baseline on CIFAR-10, without using any pre-trained models; and for ImageNet, we use the training methods proposed by [2], including using PReLU as activation function, reverse-order initialization, and knowledge distillation.

**Dependencies**

- Python 3.7
- Pytorch == 1.3.0

For the GPUs, we use a single NVIDIA RTX 2070 when training SD-BNN on the CIFAR-10 dataset and 2 NVIDIA RTX Titan when training SD-BNN on the ImageNet dataset.

**Accuracy** 

CIFAR-10:

|   Model   | Bit-Width (W/A) | Accuracy (%) |
| --------- | --------------- | ------------ |
| ResNet-20 | 1 / 1           | 86.9         |
| VGG-Small | 1 / 1           | 90.8         |
| ResNet-18 | 1 / 1           | 92.5         | 

ImageNet:

|   Model   | Bit-Width (W/A) | Top-1 (%) | Top-5 (%) |
| --------- | --------------- | --------- | --------- |
| ResNet-18 | 1 / 1           | 66.5      | 86.7      |
| ResNet-18 | 32 / 1          | 67.6      | 87.4      |

**Reference** 

[1] Haotong Qin, Ruihao Gong, Xianglong Liu, Mingzhu Shen,
Ziran Wei, Fengwei Yu, and Jingkuan Song. Forward and
backward information retention for accurate binary neural networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 2247-2256, 2020.

[2] Adrian Bulat, Georgios Tzimiropoulos, Jean Kossaifi, and
Maja Pantic. Improved training of binary networks for human pose estimation and image recognition. CoRR, abs/1904.05868, 2019.

## Citation

If you find our code useful for your research, please consider please consider citing:

    @article{xue2022self,
      title={Self-distribution binary neural networks},
      DOI={10.1007/s10489-022-03348-z},
      author={Xue, Ping and Lu, Yang and Chang, Jingfei and Wei, Xing and Wei, Zhen},
      journal={Applied Intelligence},
      year={2022},
      month={Feb}
    }

