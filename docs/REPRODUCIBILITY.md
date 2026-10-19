# 环境与复现

## 环境

当前 Windows 工作区用 `.venv_tcn/Scripts/python.exe`，Python 3.13.4。主要已安装版本记录于根目录 `requirements-core.txt`；其中 PyTorch 是本机 CPU wheel 2.9.1+cpu。MATLAB/Octave 历史分析和第三方源码各有独立依赖，`requirements-core.txt` 只覆盖当前 Python 闭环主线，不能当作所有历史脚本的完整锁文件。

仓库已初始化为本地 Git 项目。`.gitignore` 排除 `Data/`、`external/`、虚拟环境及大型模型数组；因此仅克隆代码快照不会自动获得公开数据、第三方仓库或既有仿真工件。公开下载链接和最小重建顺序见[下载清单](DATA_DOWNLOADS.md)。

如需重建环境，先创建 Python 3.13 虚拟环境，再安装匹配机器的 PyTorch wheel 及 `requirements-core.txt`；随后运行快速检查。不要将 `.venv_tcn`、`Data`、大 `.mat/.npz` 或浏览器 profile 提交到普通 Git 仓库。

## 复现顺序

1. **克隆后源码检查（不需数据）**：`python scripts\project_check.py --source-only`
2. **完整工件检查（不积分方程）**：`& .\.venv_tcn\Scripts\python.exe scripts\project_check.py`
3. **已有队列的封存协议评分（秒级）**：`& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\blind_closed_loop_benchmark.py`
4. **逐肌节时变协议重算（约数分钟）**：`& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\sarcomere_protocol_suite.py`
5. **原始单脉冲求解与消融**：`& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\sarcomere_closed_loop.py`
6. **从头生成 51 状态队列**：`run_embodied_cohort.py` 需分别用既有源、确认 manifest 对应种子和输出目录；这是较贵步骤，请先阅读 `analysis/shape_feedback/embodied_trajectory_space_report_2026-10-18.md`。

`project_check.py --deep` 额外重算轻量评分并检验结果结构。脚本入口均以项目根目录为工作目录。评测文件中源队列、确认队列、校准和封存协议分开标注；禁止用封存响应调整模型再作为同一测试结果汇报。

## 对比真实数据时的边界

公开 ACTN2-Citrine 影片只验证真实肌节运动范围和时序，不与本项目模型细胞配对。真实 5 kPa 基底刚度与当前模型 -5 无量纲预应变或 12 kPa 外加应力均不是相同控制量。任何真实细胞精度声明都需要同一细胞、同一刺激的同步电/钙/机械观测和事先冻结的参数映射。

