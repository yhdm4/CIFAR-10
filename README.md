# CIFAR-10 图像分类实验

<<<<<<< HEAD
本项目使用 PyTorch 在 CIFAR-10 数据集上进行图像分类实验，包含两个训练脚本：

- `CIFAR-10_CNN.py`：自定义卷积神经网络。
- `CIFAR-10_ResNet18.py`：基于 torchvision 预训练 ResNet-18 的迁移学习模型。

## 项目结构

```text
.
├── CIFAR-10_CNN.py          # 自定义 CNN 训练与测试脚本
├── CIFAR-10_ResNet18.py     # ResNet-18 迁移学习训练与测试脚本
├── result.md                # 已记录的 ResNet-18 实验输出
├── data/                    # CIFAR-10 数据集目录，未纳入 Git
└── README.md
```

## 环境要求

建议使用 Python 3.9 或更高版本，并安装以下依赖：

```bash
pip install torch torchvision
```

如果需要使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch 版本。安装方式可参考 PyTorch 官方安装命令。

## 数据准备

两个脚本都从 `./data` 目录读取 CIFAR-10 数据集，并且当前代码中 `download=False`，因此需要提前准备好数据：

```text
data/
└── cifar-10-batches-py/
```

如果本地还没有数据集，可以将脚本中的 `download=False` 临时改为 `download=True`，首次运行时由 torchvision 自动下载。

## 运行方式

运行自定义 CNN：
=======
本项目基于 CIFAR-10 数据集完成图像分类实验，主要对比两种方法：

1. 自定义 CNN 模型
2. 基于 ResNet18 的迁移学习模型

项目目标是理解卷积神经网络在图像分类任务中的基本流程，并比较普通 CNN 与预训练 ResNet18 在 CIFAR-10 分类任务上的表现差异。

---

## 1. 项目简介

CIFAR-10 是一个经典的图像分类数据集，共包含 10 个类别：

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

每张图片大小为：

```text
32 × 32 × 3
```

本项目使用 PyTorch 实现图像分类模型，并完成训练、测试和结果记录。

---

## 2. 项目结构

```text
CIFAR-10/
├── CIFAR-10_CNN.py          # 自定义 CNN 模型训练代码
├── CIFAR-10_ResNet18.py     # 基于 ResNet18 的迁移学习训练代码
├── result.md                # 实验结果记录
├── README.md                # 项目说明文档
└── .gitignore
```

---

## 3. 实验方法

### 3.1 自定义 CNN 模型

`CIFAR-10_CNN.py` 中实现了一个基础卷积神经网络，用于完成 CIFAR-10 图像分类任务。

CNN 的基本结构包括：

```text
输入图像
↓
卷积层
↓
激活函数 ReLU
↓
池化层
↓
全连接层
↓
分类输出
```

CNN 模型通过卷积层提取图像的局部特征，例如边缘、纹理和形状信息，再通过全连接层完成最终分类。

---

### 3.2 ResNet18 迁移学习模型

`CIFAR-10_ResNet18.py` 中使用 ResNet18 进行迁移学习。

ResNet18 是一种经典的残差网络，其核心思想是引入残差连接：

```text
output = F(x) + x
```

残差连接可以缓解深层神经网络训练中的梯度消失问题，使模型更容易训练。

在迁移学习实验中，使用预训练的 ResNet18 作为特征提取器，并将最后的分类层修改为适配 CIFAR-10 的 10 分类任务。

---

## 4. 数据集说明

本项目使用 CIFAR-10 数据集。

数据集包含：

```text
训练集：50000 张图片
测试集：10000 张图片
类别数：10 类
图片大小：32 × 32
```

每个类别包含相同数量的图像样本。

---

## 5. 环境依赖

建议使用以下环境：

```text
Python >= 3.8
PyTorch
torchvision
numpy
matplotlib
```

安装依赖：

```bash
pip install torch torchvision numpy matplotlib
```

如果使用 Anaconda，也可以创建独立环境：

```bash
conda create -n cifar10 python=3.9
conda activate cifar10
pip install torch torchvision numpy matplotlib
```

---

## 6. 运行方法

### 6.1 训练自定义 CNN
>>>>>>> f748b7415669a870d42cad1081b54dc3acfcdc29

```bash
python CIFAR-10_CNN.py
```

<<<<<<< HEAD
运行 ResNet-18：
=======
### 6.2 训练 ResNet18 迁移学习模型
>>>>>>> f748b7415669a870d42cad1081b54dc3acfcdc29

```bash
python CIFAR-10_ResNet18.py
```

<<<<<<< HEAD
脚本会自动检测设备：

- 有可用 CUDA 时使用 GPU。
- 否则使用 CPU。

训练结束后会在终端输出测试集准确率。

## 模型说明

### 自定义 CNN

`CIFAR-10_CNN.py` 使用三组卷积模块，每组包含卷积、BatchNorm、ReLU 和池化操作，最后通过全局平均池化、Dropout 和全连接层输出 10 个类别。

主要训练配置：

- 输入尺寸：`32 x 32`
- Batch size：`128`
- Epochs：`50`
- 损失函数：`CrossEntropyLoss(label_smoothing=0.1)`
- 优化器：`Adam(lr=0.001, weight_decay=1e-4)`
- 学习率调度：`CosineAnnealingLR`
- 数据增强：随机裁剪、随机水平翻转、标准化

### ResNet-18

`CIFAR-10_ResNet18.py` 使用 `torchvision.models.resnet18` 的 ImageNet 预训练权重，并将最后的全连接层替换为 10 分类输出层。

主要训练配置：

- 输入尺寸：Resize 到 `224 x 224`
- Batch size：`128`
- Epochs：`50`
- 损失函数：`CrossEntropyLoss(label_smoothing=0.1)`
- 优化器：`Adam(lr=0.001, weight_decay=1e-4)`
- 学习率调度：`CosineAnnealingLR`
- 数据增强：随机水平翻转、标准化

## 实验结果

`result.md` 中记录了 ResNet-18 的两次训练结果：

| 模型 | 说明 | 测试准确率 |
| --- | --- | --- |
| ResNet-18 | 首次运行，下载预训练权重后训练 | `87.94%` |
| ResNet-18 | 再次运行，使用本地缓存的预训练权重 | `94.28%` |

实际结果会受到随机初始化、数据增强、硬件环境和依赖版本影响。

## 注意事项

- `data/` 已在 `.gitignore` 中忽略，数据集不会提交到仓库。
- ResNet-18 首次运行时需要下载预训练权重，如果网络不可用，需要提前准备好权重缓存。
- 当前脚本只打印训练损失和最终测试准确率，不会自动保存模型权重。
=======
运行后，程序会自动下载 CIFAR-10 数据集，并开始模型训练。

---

## 7. 实验结果

实验结果记录在 `result.md` 文件中。

可以记录以下指标：

| 模型 | 训练方式 | 测试准确率 | 备注 |
|---|---|---:|---|
| 自定义 CNN | 从零训练 | 待填写 | 基础卷积神经网络 |
| ResNet18 | 迁移学习 | 待填写 | 使用预训练模型 |

从实验结果可以观察：

1. 自定义 CNN 能够学习 CIFAR-10 的基本图像特征。
2. ResNet18 由于具有更深的网络结构和预训练参数，通常能够取得更好的分类效果。
3. 迁移学习可以减少训练难度，并提升模型在小规模数据集上的表现。

---

## 8. 模型对比分析

### 自定义 CNN 的特点

优点：

```text
结构简单
便于理解卷积神经网络原理
训练和调试方便
```

缺点：

```text
模型表达能力有限
对复杂图像特征的提取能力较弱
准确率通常低于深层网络
```

### ResNet18 的特点

优点：

```text
网络更深
残差结构更容易训练
预训练模型具有较好的特征提取能力
分类效果通常更好
```

缺点：

```text
模型参数更多
训练开销更大
结构理解难度高于普通 CNN
```

---

## 9. 项目总结

本项目完成了 CIFAR-10 图像分类实验，并对比了自定义 CNN 和 ResNet18 迁移学习方法。

通过本实验，可以掌握：

1. CIFAR-10 数据集的基本使用方法。
2. PyTorch 图像分类任务的完整训练流程。
3. CNN 的基本结构和工作方式。
4. ResNet18 残差网络的基本思想。
5. 迁移学习在图像分类任务中的应用。

整体来看，普通 CNN 适合用于理解基础原理，而 ResNet18 更适合用于提升分类效果。

---
>>>>>>> f748b7415669a870d42cad1081b54dc3acfcdc29
