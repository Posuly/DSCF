
# DSCF: Disentangled Invariant Style-Content Framework

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of **DSCF** (Disentangled Invariant Style-Content Framework). 


## ⚙️ 核心架构 (Architecture)
框架的设计包含以下模块：
1. **HVA**: 负责处理异构流量特征。
2. **NE**: 针对网络环境/噪声特征的处理。
3. **CS-DE (Content-Style Disentanglement Encoder)**: 内容-风格解耦编码器，实现环境依赖和应用本体特征的分离。
## 🛠️ 环境依赖 (Requirements)
请确保你的环境中安装了以下基础库（推荐使用 Anaconda 管理环境）：
```bash
python >= 3.8
torch >= 1.12.0
numpy
scikit-learn
pandas
