# 力电耦合虚拟细胞研究项目

本项目研究机械干预如何影响单细胞电活动、钙动态和收缩，以及少量观测能否支持对**未见干预**的预测。代码、公开数据和逐轮数值审计保留在同一工作区。当前主平台是人源 ToR-ORd/SAC–Land 电/钙模型与内生力学状态耦合的**计算原型**；尚无同一真实细胞同步 AP/Ca/受控力/逐肌节形变数据完成闭环验证。

## 从这里开始

| 目的 | 入口 |
|---|---|
| 当前科学结论、误差与物理边界 | [`analysis/shape_feedback/action_conditioned_virtual_cell_report.md`](analysis/shape_feedback/action_conditioned_virtual_cell_report.md) |
| 全项目已确认结果与证据层级 | [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) |
| 可运行的 51 状态外应力闭环与两批虚拟细胞 | [`analysis/shape_feedback/README.md`](analysis/shape_feedback/README.md) |
| 90 状态逐肌节闭环、消融及数值审计 | [`analysis/shape_feedback/sarcomere_closed_loop_report.md`](analysis/shape_feedback/sarcomere_closed_loop_report.md) |
| 离线交互式仿真展示 | [`analysis/shape_feedback/sarcomere_closed_loop_viewer/index.html`](analysis/shape_feedback/sarcomere_closed_loop_viewer/index.html) |
| 代码、数据、旧实验如何对应 | [`docs/PROJECT_MAP.md`](docs/PROJECT_MAP.md) |
| 数据来源、存储与清理原则 | [`docs/DATA_AND_STORAGE.md`](docs/DATA_AND_STORAGE.md) |
| 公开数据、原作者模型下载清单 | [`docs/DATA_DOWNLOADS.md`](docs/DATA_DOWNLOADS.md) |
| 环境、复现和快速检查 | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| 后续实验与代码开发约定 | [`docs/DEVELOPMENT_RULES.md`](docs/DEVELOPMENT_RULES.md) |

## 当前主线

```text
机械输入 / 起搏
       ↓
内生长度与逐肌节运动 ↔ 电活动 V ↔ Ca ↔ 主动张力
       ↓
基线与少量校准观测 → 封存机械协议的 V/Ca/张力/长度轨迹评测
```

闭环方程能运行，原单脉冲回归和求解器收敛已通过；时变协议评测显示多数平滑响应可由简单因果基线预测，少数事件型电活动仍会失败。**数值正确性、同方程预测、真实细胞验证是三个不同层级**。细胞膜与通道的可视化位置是示意，不是空间分辨的真实重建。

## 快速检查

GitHub 克隆后先运行无需数据的源码检查。完整数值检查需要按[下载清单](docs/DATA_DOWNLOADS.md)放置第三方模型并生成对应队列。

```powershell
python scripts\project_check.py --source-only
```

在**已有本地数据和仿真队列**的工作区，使用现有环境：

```powershell
& .\.venv_tcn\Scripts\python.exe scripts\project_check.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\blind_closed_loop_benchmark.py
```

第二条命令重算轻量的 24 细胞封存协议评分；不重新运行昂贵的 ODE 队列。重跑完整逐肌节协议见[复现说明](docs/REPRODUCIBILITY.md)。项目目录未强行搬迁历史脚本，因为许多分析与原始数据路径有明确依赖；新增开发应先阅读项目地图，避免复用旧结果作为新测试集。

## 数据与 GitHub

`Data/`、`external/`、虚拟环境、原始显微电影、模型权重及大型 `.mat/.npz` 仿真输出由 `.gitignore` 排除；仓库保留生成代码、小型指标表和论文图，方便审计。主要下载入口：[Duke AFM–电流 Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.gtht76hq5)、[Strocchi 仿真特征 Zenodo](https://zenodo.org/records/7405335)、[Härtter 肌节轨迹 Zenodo](https://doi.org/10.5281/zenodo.17564384)、[TTM AP/Ca/收缩特征 Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.67c92vm)。这些资源来自不同实验/模型体系，按需下载和分别分析；具体文件与本地路径见[完整清单](docs/DATA_DOWNLOADS.md)。

