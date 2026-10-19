# 细胞级机械转导表型：第一轮可证伪结果

分析日期：2026-09-27。主队列沿用已经冻结的 −80±5 mV、带符号电流、响应与 SNR QC，AFM 力峰对齐，以及 1 ms 波形采样。每个细胞只占一行；同一细胞的合格 sweep 用中位数汇总。chimA 和其他扰动没有参与 WT 模型的特征标准化、权重或参数选择。这是**回顾性表型**，不是未来电流波形的前瞻预测。

## 数据与特征

WT Piezo1 11 个细胞、8 个日期、27 条合格 sweep；WT Piezo2 11 个细胞、3 个日期、34 条 sweep；Piezo1^cap2（原始表中 `chimA`）11 个细胞、2 个日期、11 条 sweep。机械刺激速度均为作者表中的 40；电压限制由冻结波形 QC 实施。原始数据与 chimera 的生物学身份参见 [Young 等原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9974467/)。

主轴使用三个细胞级数值：log 工作阈值、AFM 力峰到电流峰的延迟、峰后 20 ms 带符号电流/峰值。没有使用原始电流幅度。这个主轴是在发现 exponent 的拟合范围问题后确定的，因此本轮检验属于**探索性模型选择**。对于每个留出日期，只用其余日期拟合标准化和固定 L2 惩罚的线性 logistic 模型；训练权重使每个通道内各日期等权。最后再用所有 WT 冻结一个供 chimA 投影的模型。轴上的 0、1 分别定义为该训练模型中 WT Piezo1、Piezo2 的细胞中位数。

另计算单细胞 work–current log-log 指数，拟合规则与既有 `within_cell_power.py` 相同，chimA 从其原始 `capchims_approaches.csv` 复算。该指数在审计后**没有进入主轴**。

## 1. WT 跨日期检验

| 本轮比较的特征集 | 留日期细胞 AUC | 细胞平衡准确率 | 日期平衡准确率 |
| --- | ---: | ---: | ---: |
| 主轴：阈值 + 延迟 + 20 ms 保留比例 | 0.975 | 0.955 | 1.000 |
| 仅延迟 | **0.992** | 0.909 | 0.875 |
| 阈值 + 保留比例，不含延迟 | 0.744 | 0.682 | 0.875 |
| 四特征，额外加入 work exponent | 0.967 | 0.909 | 1.000 |

主轴对未见日期的 22 个细胞分组有较好分数；11 个留出日期的平均分数都位于正确一侧。把三组 Piezo2 日期标签放在 11 个日期中的所有 165 种可能位置时，有 3 种得到至少同样高的日期 AUC（经验比例 0.018；特征集经本轮审计选择，不能当作预注册显著性检验）。**这不是批次校正后的通道效应检验**：WT Piezo1 与 Piezo2 的采集日期完全不重叠，任何留日期结果都不能独立排除通道和日期相关的机械/记录差异。更关键的是，单独用延迟的 AUC 甚至高于多特征主轴，所以目前没有证据说 WT 区分必须由一个新的多维潜变量完成。

## 2. 冻结 WT 轴后检验 chimA

在冻结主轴上，chimA 的细胞中位位置为 **0.995**（WT Piezo1=0、WT Piezo2=1）。仅延迟训练得到的轴也给出 chimA=1.000；因此这个位置本身不能算作超越已知 cap-domain 延迟现象的新发现。原论文独立报告 Piezo1^cap2 的延迟接近 Piezo2，工作阈值也向 Piezo2 移动。[Young 等原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9974467/)

真正值得继续的是**各观测量移动得不一样**：

| 每细胞中位数 | WT Piezo1 | WT Piezo2 | chimA | 当前解释 |
| --- | ---: | ---: | ---: | --- |
| 工作阈值，fJ | 159.4 | 114.8 | **91.5** | chimA 在本队列落在更敏感的一侧；日期混杂仍在 |
| 力峰→电流峰延迟，ms | 8.0 | 2.0 | **2.0** | 与原论文一致，Piezo2-like |
| 电流峰后 20 ms 保留比例 | 0.506 | 0.402 | **0.186** | 比两个 WT 组都低；两个 chimA 日期的中位数分别约 0.190、0.179 |
| 原始拟合 work exponent | 5.15 | 2.84 | 4.27 | **不用于机制结论**，对拟合工作范围敏感 |

在 chimA，电流峰后 20 ms 的 AFM 力保留比例中位数为 **0.772**，电流为 **0.186**；WT Piezo2 分别为 **0.824**、**0.402**。三个组各有 11/11 细胞在这个窗口的电流下降快于测得的全局 AFM 力。这给“峰后持久性”一个比原始电流衰减更具体的表型对照，但阈值非线性、局部张力松弛、细胞骨架与通道门控仍可能共同造成差异，不能把它命名为 cap 引起的本征失活常数。chimA 仅有两个独立实验日期，且 WT 无同日对照。

## 3. 为什么 work exponent 暂缓作为表型维度

WT Piezo1 的 11 个细胞中有 5 个用于指数拟合的 log10-work 范围小于 0.2；WT Piezo2 为 0/11；chimA 为 1/11。WT 中指数与该范围呈明显负相关。把所有 WT 的指数对范围做**诊断性**线性调整到共同范围后，Piezo1/Piezo2 中位数分别约 **4.55/4.51**，原始的 5.15/2.84 差异几乎消失。此调整不是新的生物物理模型，但足以说明原始指数差异不能直接解释为通道特异非线性。四特征模型也没有改善留日期表现。指数可以留在审计表中，不宜成为当前论文主张。

## 4. 外部机制证据的正确位置

已复算的 [PIEZO2–FLNB Source Data](piezo2_flnb_source_data_report.md)显示，FLNB KO 使 Piezo2 压入阈值均值从 3.34 增至 4.71 µm，峰电流密度均值降至对照约 0.49 倍；相同体系的 Piezo1 没有相应稳定变化。[原论文](https://www.nature.com/articles/s41586-026-10182-7) 已复算的 [PIEZO1 wedge Source Data](wedge_source_data_report.md)显示，50 ms 脉冲间隔下第六脉冲/第一脉冲中位数 WT 0.733、wedgeDEL 0.991。[原论文](https://www.nature.com/articles/s41467-026-76927-0) 这些实验分别为**受力路径**与**历史响应**提供独立分子扰动证据；所测机械量、细胞背景、刺激方式不同，不能把它们投到 Duke 的同一个数值坐标。最新 [Piezo2 剪接研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC13382605/)也支持张力敏感性和压入敏感性会随通道结构改变；因此 Piezo2 的机械编码不宜写成绝对的“只靠 tether”。

## 当前可守住的表述与下一道门槛

最稳妥的正结果是：**在严格 QC 的 Duke 子队列中，cap2 chimera 的响应时序落在 Piezo2 一侧，同时工作阈值与峰后持久性呈不等幅的变化。** 峰后电流下降比同期测得的全局 AFM 力更快。这个结果是一个可检验的*模块化表型指纹*，尚不是已证实的通道无关低维状态或分子机制分解。

论文主张要升级为“扰动选择性重构多维机械转导表型”，需要同日、同协议的 WT P1/P2/chimA 对照，并增加随机化机械输入。尤其应预先指定阈值、延迟、力校正后的 20 ms 保留比例；新的功率指数须先解决拟合范围差异。现有数据适合据此设计实验和形成 pilot figure，不足以作最终因果归因。

## 复现文件

运行 `.venv_tcn/Scripts/python.exe analysis/mechanotransduction_phenotype.py`，随后运行 `.venv_tcn/Scripts/python.exe analysis/phenotype_force_retention_audit.py`。结果见 `mechanotransduction_phenotype_results.json`、`mechanotransduction_phenotype_cells.csv`、`mechanotransduction_phenotype_dates.csv`、`mechanotransduction_phenotype_lodo.csv`、`mechanotransduction_phenotype_projection.csv`、`phenotype_force_retention_results.json`；可视化为 `mechanotransduction_phenotype.png` 和 `mechanotransduction_phenotype_holdout.png`。所有文件在本目录。
