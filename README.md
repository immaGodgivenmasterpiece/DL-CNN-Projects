# DL-CNN-Projects

A progressive series of CNN projects built with **PyTorch**, going from scratch implementations to real-world datasets.  
Each project is self-contained with its own notebook, scripts, results, and README documenting design decisions.

---

## Project Series

| # | Project | Dataset | Best Accuracy | Status |
|---|---|---|---|---|
| 1 | [MNIST CNN](./1_MNIST_CNN/) | MNIST (handwritten digits) | 99.35% | ✅ Done |
| 2 | [CIFAR-10 CNN](./2_CIFAR10_CNN/) | CIFAR-10 (10 object classes) | 81.04% | ✅ Done |
| 3 | [Sports Classification](./3_Transfer_Learning/) | Sports (100 classes, Kaggle) | — | 🔧 In Progress |

---

## Learning Progression

```
1_MNIST_CNN          → CNN basics, PyTorch training loop, Early Stopping
2_CIFAR10_CNN        → Color images, Batch Normalization, LR Scheduler (scratch CNN ceiling: ~81%)
3_Transfer_Learning  → Transfer Learning, Data Augmentation, 100-class sports dataset
```

Each project answers:
- Why did I choose this model?
- What problem did I find?
- How did I improve it?

---

## Skills Covered

- Custom CNN architecture design
- PyTorch `nn.Module`, `DataLoader`, `transforms`
- Training loop with Early Stopping & best model checkpointing
- Loss/Accuracy curve visualization
- Confusion Matrix & wrong-prediction analysis
- Transfer Learning (ResNet, EfficientNet)
- Real dataset pipeline (Kaggle API, custom `Dataset` class)

---

## Background

Built after reading:
- *Deep Learning from Scratch* (DLFS) Ch. 1–7
- *Hands-On Machine Learning* (HOML) up to the Deep Learning chapters
