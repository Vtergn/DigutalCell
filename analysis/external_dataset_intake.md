# 新增公开数据审计：ETH、eLife 与 Southampton

日期：2026-09-26。本文件记录当前 `Data` 下**实际已下载**的内容、可识别的观测层，以及与 Duke AFM–patch-clamp 研究的接口。没有把不同实验的力、张力、电流或钙信号直接拼接成同一训练集。

## 1. ETH Zürich / Lüchtefeld 等，Nature Methods 2024

本地已有 `Data/CalciumImaging.zip`（压缩约 3.51 GB）、`Data/RealtimeFlipperTrImaging.zip`（约 1.65 GB）和 `Data/StepwiseFlipperTrImaging.zip`（约 4.52 GB），均已展开。CalciumImaging 内容为：

- `SourceData_Fig3bc+Fig5ab+SI6b.xlsx`：4 个工作表；其中 `Critical Aspiration Pressures` 有 **484 个数据行**，字段为 condition、独立重复编号、压入力 nN、临界吸力 mbar。按条件分别为 Control 281、margaric acid 116、CytoD 87 行。论文主分析的 **399 个细胞**可由预设的主图压入力集合复现：Control 220、MargAc 106、CytoD 73；其余 85 行是更高压入力的补充范围。另有逐力档的累积分布和 15 个 `pC,50` 汇总拟合点。
- 两个代表性 `OME-TIFF` 钙成像视频：intact HFF 512×512、7,835 frame；broken cell 256×256、4,228 frame。它们不是 484 个细胞的完整视频集。

一个简单核对：在 2 nN 下，Control/MargAc/CytoD 的临界吸力中位数分别为 175/100/175 mbar，样本数 60/25/19；25 nN 下分别为 50/0/25 mbar，样本数 43/20/19。**0 mbar 是无附加吸力时的响应阈值，不应当当作“未响应”或缺失值。** 不同条件力档和重复分布不同，不能直接合并比较。`pC,50` 表在高压入力下还有负值，是拟合线的外推，不是实际施加了负的临界吸力。

RealtimeFlipperTrImaging 展开后有 5 个完整细胞、5 个 membrane bleb 的 `5012 × 128` 行扫描 TIFF（另有 `.lif` 原始文件和图中示例）。XML 的扫描模式为 `xt`，对应 40 ms/行、约 25.49 µm 的空间扫描长度。扫描画布标称约 200 s，但实际有荧光信号的长度在 58.0–78.0 s；其余行是零填充，必须排除。`Cell01/02` 缺相应 XML，仅能暂借同组扫描设置，具体时间轴仍需核对原始 `.lif`。导出的 XML 没有写明各 TIFF 通道的物理量或单位；因此当前仅做了原始信号的时间—空间 QC，尚未将 `ch3` 擅自标作绝对膜张力。这个包里未发现同步 AFM 力/位移数值文件，不能直接拟合逐时刻的 `F_external→T_membrane` 传递函数。[实时图像审计](eth_realtime_qc.png)；[审计数值](eth_realtime_audit.json)。

**StepwiseFlipperTrImaging.zip** 现已下载并分析：5 个压入力细胞、4 个吸力细胞的 147 帧导出 TIFF，另有 `.lif`。配合论文官方 Figure 2 源表，同细胞平均局部张力变化斜率估为压入力 **0.00698 mN/m/nN**、吸力 **0.00238 mN/m/mbar**，与原论文的 0.0065 和 0.0024 接近。原图导出 XML 缺通道单位、剂量逐帧映射和同步 AFM 数值，因此 TIFF 的定量 ROI 暂未独立复现；完整方法和不确定性见 [Stepwise 分析](eth_stepwise_report.md)。CalciumImaging 验证的是**力 + 吸力→细胞群体钙响应阈值**。论文说明完整的钙成像视频集需要向作者索取。[数据仓库](https://www.research-collection.ethz.ch/handle/20.500.11850/659903)；[论文](https://www.nature.com/articles/s41592-024-02277-8)。

将 Stepwise 力/吸力张力斜率之比固定为 **2.94 mbar/nN**，在 CalciumImaging 的 Control 细胞做独立重复留出，Brier 为 **0.0857**；钙数据内部重新学习该比例为 **0.0849**，仅用吸力为 **0.1262**。这一外部固定比例的预测接近自由拟合，但比例的小样本不确定性大，不代表唯一的膜张力机制已被识别。[交叉检验](eth_stepwise_report.md)。

实验尺度限制：论文实时 Flipper-TR 的张力 kymograph 时间分辨率为 **20–80 ms**；即使另两个包下载齐，也不足以直接解释 Duke 的 1–10 ms Piezo 电流起始延迟。Flipper-TR 荧光寿命需要在同一 HFF/探针条件下校准；论文给出的是该系统的转换与几何系数，不能把其 nN→mN/m 斜率当作 Duke AFM 的通用转换常数。

## 2. eLife 2021 / Piezo1 cell-attached patch tension gating

本地有 Figure 4、Figure 5 两份 source-data XLSX。Figure 5D/5E 共享同一列 channel density，按行能对齐 **25 个 patch** 的半最大张力 `T50` 和 slope `k`。从已下载数值直接计算：`T50` 中位 **2.487 mN/m**、四分位 **2.196–2.856 mN/m**、范围 **1.447–3.556 mN/m**；`k` 中位 **0.681 mN/m**、四分位 **0.514–0.777 mN/m**、范围 **0.164–1.213 mN/m**。这些是构建宽先验或敏感性范围的材料，比只用论文中单一的 2–3 mN/m 点估计稳妥。Figure 5F 为按通道密度分组的归一化张力–电流曲线，Figure 4 含压力敏感性、通道数量分箱与归一化曲线。

数据质量提示：Figure 5C 的第二列标为 `Current (pA)`，数值却在 `10^-11` 量级；按原表标签直接当 pA 明显不合理，可能是以 A 存储的电流。使用该列前应核实单位，不从它直接估算单通道电流。Figure 4C 的本地分箱数相加为 143，而图文提到 144 patches；若需要精确通道数量先验，应核对该差异。Figure 5D/5E 没有公开显式 patch ID，虽然 density 列逐行一致，也不把这些行与其他工作表强行联结。[论文与 Figure 4/5 数据入口](https://elifesciences.org/articles/70988/figures)。

物理解释：这是 **cell-attached 膜 patch 的压力刺激**，张力来自 patch 半径与 Laplace law，不是 Duke 的完整细胞 AFM 压入。`T50/k` 可以约束 `T_eff→P_open` 的可能量级，但不能作为 Duke 细胞的已知真实门控参数；压力–电流曲线也不是对未来 350 ms 电流波形的监督数据。

## 3. Southampton D3151 / HL60 电–机械表型

本地 `Dataset_supporting_the_publication` 已包含 README 和 Figure 1–6、补充图 2–5 的 **11 份 XLSX**。Figure 1 两个工作表的标称范围各到 16,384 行，但实际只有约 **4,418/4,417 个非空行**；它们是图中电直径–电变形/opacity 的图点来源，不能把标称行数当样本量。Figure 3–5 提供渗透、戊二醛、CytoD 等条件的图表数据。Figure 5 的 `Information` 指明部分 A/B 与 C/D 列是 beads 与 cells 的各自 x/y 图点，不应误认为同一细胞的完整多模态原始记录。Figure 6 有 **3 个重复的群体均值**，不能把图点数当成独立生物重复数。某些源表工作表标题沿用 `Figure 4`，即使文件名为 `Figure5.xlsx`，提取时应以 README/Information 和具体列说明为准。

论文设备确实测量正交电极与两频阻抗，但下载的图表源数据主要是已计算的 electrical deformability（低频 `Z1/Z2`）、electrical diameter、opacity 及对应光学表型；未发现可直接用于逐细胞时序仿真的完整原始 `Z1(t), Z2(t)` 或频谱。HL60 悬浮细胞在流动剪切下的表型也不能直接作为 Duke HEK293T Piezo 电流的训练标签。适合做**独立的工程测量/细胞骨架扰动表型分支**，优先级低于 ETH Flipper 数据和 eLife 张力先验。[数据集](https://eprints.soton.ac.uk/495905/)；[论文](https://www.nature.com/articles/s41378-024-00810-5)。

## 对当前项目的具体接法

1. **现在可做**：将 eLife Figure 5D/5E 的 25 个 patch 分布用于 `T_eff→P_open` 的宽先验与敏感性分析；将 ETH 399 个主分析细胞的临界吸力按 condition、repeat、force 做**外部群体阈值检查**。这些观测不应被当成 Duke 的额外训练细胞。
2. **ETH 两个 FLIM 包已到**：Stepwise 加论文官方源表可约束同一 HFF/FluidFM 几何内的静态 `F_external→ΔT_membrane` 斜率；Realtime 可进一步检查完整细胞与 bleb 的空间传播，但定量寿命动态仍须确认通道定义、膜区 ROI 和刺激同步。映射到 Duke 时需保留未知几何比例、细胞类型和时间分辨率限制。
3. **Southampton 放第二阶段**：在单独任务里研究剪切变形与阻抗表型及 CytoD 效应，用于扩展“electromechanical phenotype”，不让它替代 Piezo patch-clamp 电流。

当前最重要的修正是：**Realtime 包提供局部张力成像的原始材料，但仅靠其导出 TIFF/XML 和当前的同步信息，尚不能获得已校准的力→张力动态传递函数。**
