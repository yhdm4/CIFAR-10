# CIFAR-10 图像分类实验

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

```bash
python CIFAR-10_CNN.py
```

运行 ResNet-18：

```bash
python CIFAR-10_ResNet18.py
```

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
