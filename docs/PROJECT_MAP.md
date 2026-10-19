# 项目地图

各分支的结果数字、GO/NO-GO 和证据边界集中于[研究结果索引](RESULTS_INDEX.md)。本页主要帮助定位代码和输入。

## 活跃模块

| 模块 | 代码与输入 | 权威结果 / 限制 |
|---|---|---|
| 人源 50 状态电–钙–Land 核心 | `analysis/torch_native_port/`；ToR-ORd、Land 来源见该目录 README | 原方程移植、数值轨迹审计；不包含真实细胞个体化 |
| 51 状态外应力闭环 | `analysis/shape_feedback/embodied_length_pilot.py`、`run_embodied_cohort.py` | `embodied_cohort/` 与 `embodied_cohort_confirm/` 两批各 12 细胞；外层刚度、黏性是探索性设定 |
| 51 状态封存协议测试 | `analysis/shape_feedback/blind_closed_loop_benchmark.py` | `blind_closed_loop/results.json` 和 `action_conditioned_virtual_cell_report.md`；只用基线和校准轨迹预测两个目标 |
| 90 状态逐肌节闭环 | `analysis/shape_feedback/sarcomere_closed_loop.py`、`sarcomere_stochastic_cosim.py` | `sarcomere_closed_loop_report.md`；原作者肌节力–速度代码来自 `external/SarcomereModel`，桥接比例尺未标定 |
| 90 状态时变协议 | `analysis/shape_feedback/sarcomere_protocol_suite.py` | `sarcomere_protocol_suite/results.json`；同一名义细胞，不能据此推断群体性能 |
| 真实肌节轨迹与展示 | `analysis/shape_feedback/extract_sarcomere_tracks.py`、`cell_visualization/`、`sarcomere_closed_loop_viewer/` | 影像与模拟不是配对的同一细胞/刺激；外观、通道位置为示意 |

## 已完成的研究分支

- **Duke Piezo 力电数据**：`Data/2022_pnas/`；分析集中于 `analysis/` 顶层脚本和 `duke_raw_cleanup_report.md` 等报告。它与人源心肌模型是不同体系，不能合并为同一训练表。
- **Strocchi/ToR-ORd–Land 合成队列**：`analysis/virtual_cardiomyocyte_5000/`、`analysis/virtual_cell_benchmark_report.md` 和相关报告。可用于单向 AP/Ca→张力、参数可观测性与负载外推；不是已验证的双向真实细胞数据。
- **兔心肌 MCT2024 闭环参考**：`analysis/mct_feedback_audit/` 与 `analysis/mct_*_report.md`。物种与方程均不同，不能把其参数当作人源模型真值。
- **MechChem 跨模型检验**：`analysis/mechchem_cross_model/` 与 `analysis/cross_model_static_geometry_report.md`。较大的 `.mat` 是原始运行产物，保留用于追溯。
- **历史模型去留试验**：`analysis/` 中 CNN、ODE、UDE、主动设计、流形等脚本和对应报告。它们记录失败与边界，不作为当前闭环平台的默认训练入口。

## 依赖方向

`Data/` 是外部资料和用户下载数据；`external/` 是第三方代码；`analysis/` 是本项目实验、报告和生成产物。不要编辑外部原始数据来适配模型。若新算法依赖已有结果，记录所用队列、种子、校准协议、留出协议和代码版本。历史分析中有硬编码相对路径，因此没有批量移动文件；要重构某一模块时先迁移该模块的输入与测试。

