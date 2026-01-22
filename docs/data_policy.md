# Data & Checkpoints Policy

本仓库只存放：代码、配置、文档、小型示例。

**严禁提交到 Git 的内容：**
- 数据集（radar/gpm/station 的原始与中间产物）
- 模型权重与 checkpoint
- 训练过程产物（wandb、runs、logs、大型可视化）

推荐约定（本地使用，均应被 .gitignore 忽略）：
- 数据：./data/  或 ./datasets/
- 权重：./checkpoints/ 或 ./weights/
- 训练日志：./runs/ ./wandb/ ./logs/

如果必须共享权重：请使用对象存储 / Release Assets / 内部制品库（或经允许后使用 Git LFS）。
