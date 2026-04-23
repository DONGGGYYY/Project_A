# LIBERO-Spatial：资源受限条件下的 BC 与 Diffusion Policy 对比

[English](README.md) | [简体中文](README.zh-CN.md)

> 一个实证研究：在机器人操作模仿学习中，什么时候简单的 Behavioral Cloning 会优于 Diffusion Policy。
> 全部实验在单张 8GB 笔记本 GPU 上完成。

![热力图：chunk8](libero-bc-vs-dp-study/outputs/task_generalization_multitask_chunk8.png)

## 摘要（TL;DR）

我们在 LIBERO-Spatial 上训练了基于 ResNet18 的 **MLP BC**（11M 参数）和 **Diffusion Policy**（15M 参数，1D Conv-UNet + EMA + DDIM），
训练数据为 10 个任务 x 50 条演示，共 500 条轨迹；每组设置都做了跨任务 100 次评估（10 task x 10 ep）。
三组消融结果表明：**BC 和 DP 谁更好，取决于观测信息是否充足**。

| 设定 | MLP BC | Diffusion Policy | 赢家 |
|---|---|---|---|
| 单视角（agentview），`chunk_steps=1` | **33%** | 17% | BC +16 |
| 单视角，`chunk_steps=8`（DP 论文默认） | **33%** | 28% | BC +5 |
| **双视角（+ wrist cam），`chunk_steps=8`** | 21% | **32%** | **DP +11** |

**核心发现**：
1. **DP 对超参数高度敏感**：在模型和训练数据不变的前提下，仅通过调整 `chunk_steps` 并加入 wrist camera，成功率可从 17% 提升到 32%。
2. **加入 wrist camera 反而会伤害 MLP**（33% -> 21%），即便测试 MSE 下降了 32%（0.0226 -> 0.0154）。这说明模仿学习里常见的现象：**动作回归损失不等于任务成功率**。
3. BC 与 DP 的优劣并非绝对：观测信息弱时，确定性回归更稳；观测信息更丰富时，DP 的多模态生成优势更容易发挥。

## 为什么这个项目有意义

很多论文只在单一设定下比较 BC 与 DP。本项目显示：仅改变两个因素（`chunk_steps` 和相机视角），
同一组架构就可能发生“赢家翻转”。这对资源受限或单相机机器人场景下的模型选型非常实用。

## 仓库结构

```text
Project_A/
├── README.md
├── README.zh-CN.md
├── .gitignore
└── libero-bc-vs-dp-study/
    ├── 02_dataset.py                       # LIBERO HDF5 loader + multi-task ConcatDataset
    ├── 03_train_mlp.ipynb                  # 单任务 MLP 训练（旧）
    ├── 04_eval.py                          # Rollout 评估（自动识别 MLP / MLP+wrist / DP）
    ├── 05_diffusion_policy.py              # Diffusion Policy: 1D Conv-UNet + DDIM + EMA
    ├── 06_train_diffusion.ipynb            # 单任务 DP 训练（旧）
    ├── 07_diagnose.py                      # DP 失败模式分析（动作漂移、夹爪信号）
    ├── 08_task_generalization.py           # 跨任务评估入口
    ├── 09_train_mlp_multitask.py           # 多任务 MLP（单视角）
    ├── 09b_train_mlp_multitask_wrist.py    # 多任务 MLP（双视角 wrist 消融）
    ├── 10_train_diffusion_multitask.py     # 多任务 Diffusion Policy
    └── outputs/                            # 模型、评估 JSON、热力图
```

## 结果复现

### 环境准备

```bash
# WSL2 + Ubuntu，建议 conda 环境
conda create -n libero python=3.10
conda activate libero
cd libero-bc-vs-dp-study/LIBERO && pip install -e . && cd ../..
# 其余依赖：PyTorch 2.3.1+cu121、diffusers、h5py 等
```

数据集（LIBERO-Spatial，10 个 hdf5 文件）通过 `libero.libero.get_libero_path("datasets")` 自动定位。

### 训练

```bash
# 在 Project_A/ 目录下
cd libero-bc-vs-dp-study

# 多任务 MLP（单视角，约 22 分钟，RTX 3070 Ti 8GB）
python 09_train_mlp_multitask.py

# 多任务 MLP（双视角 wrist 消融，约 22 分钟）
python 09b_train_mlp_multitask_wrist.py

# 多任务 Diffusion Policy（约 70 分钟，15 epochs）
python 10_train_diffusion_multitask.py
```

所有训练脚本默认使用 `cache_in_ram=True`，通过将 demo 预加载到内存获得约 70x 的 I/O 加速。

### 评估

```bash
# 在 Project_A/libero-bc-vs-dp-study/ 目录下

# 单视角，DP 论文默认 chunk_steps=8
python 08_task_generalization.py \
    --mlp outputs/model_mlp_multitask.pt \
    --dp  outputs/model_diffusion_multitask.pt \
    --tag multitask_chunk8 --chunk-steps 8 --episodes 10

# 双视角消融（自动识别 wrist MLP checkpoint）
python 08_task_generalization.py \
    --mlp outputs/model_mlp_multitask_wrist.pt \
    --dp  outputs/model_diffusion_multitask.pt \
    --tag multitask_wrist --chunk-steps 8 --episodes 10
```

每条评估命令执行规模约为 10 task x 10 ep x 2 model，总耗时约 90 分钟，
会在 `outputs/` 下生成 JSON 摘要与热力图 PNG。

## 关键实现细节

- `cache_in_ram=True`（见 [libero-bc-vs-dp-study/02_dataset.py](libero-bc-vs-dp-study/02_dataset.py)）：在构造数据集时预载 500 条 demo，训练迭代速度从约 3.7s/iter 提升到约 0.05s/iter。
- DP 的 receding-horizon（见 [libero-bc-vs-dp-study/04_eval.py](libero-bc-vs-dp-study/04_eval.py)）：每次预测 16 步，仅执行前 K 步后重采样。K=1 容易破坏轨迹平滑，K=8 更接近论文设定。
- wrist 双视角输入（见 [libero-bc-vs-dp-study/09b_train_mlp_multitask_wrist.py](libero-bc-vs-dp-study/09b_train_mlp_multitask_wrist.py)）：将 `agentview_rgb` 与 `eye_in_hand_rgb` 按通道拼接（6 通道），并将 ResNet18 第一层卷积从 3 通道改为 6 通道。
- DP 训练策略：EMA + DDIM（decay=0.995，train steps=100，inference steps=10）。

## 未覆盖内容

- 语言条件策略（CLIP / T5 指令编码）
- 更大 backbone（ViT、R3M）
- 真机迁移
- 其他 LIBERO 套件（Goal / Object / 100）

## 局限性

- 表格中每个点仅 10 episodes，噪声较大（约 ±15%），小于 5% 的差异不宜过度解读。
- DP 仅训练 15 epochs（MLP 为 30），受时间预算限制；若增加训练时长，DP 结果可能继续提升。
- 单随机种子，无误差条。

## 致谢

本项目基于 [LIBERO benchmark](https://libero-project.github.io/) 与 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 的公开实现。
内部工作笔记为私有内容，不包含在公开仓库中。