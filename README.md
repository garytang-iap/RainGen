# RainGen

**RainGen aims to be a probabilistic nowcasting system with high fidelity across space (any resolution), time (minute-scale dynamics), and intensity (long-tail extremes).**

RainGen 是一个面向真实降水场景的短临预报框架工程代码库，包含：
- 雷达回波外推（Latent Diffusion + SVD-UNet，10→30）
- 雷达→降水映射（GPM-IMERG 监督 + 功率谱对齐 + Functional Diffusion）
- 站点微调（稀疏站点监督 + DreamBooth 风格微调）
- 概率预报（集合采样/不确定性刻画）

## 方法概览
详见：`docs/overview.md`

## 仓库结构
- `models/`：当前已有的模型工程目录（后续将逐步规范化整理）
- `scripts/`：训练/推理/评估脚本入口（后续统一）
- `configs/`：实验配置（后续规范化）
- `docs/`：方法与数据规范文档
- `tests/`：测试（逐步补齐）

## 数据与权重
本仓库不提交数据集与模型权重。详见：`docs/data_policy.md`
