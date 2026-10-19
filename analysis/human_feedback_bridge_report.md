# 人心肌双向反馈：公开模型核查与最小可证伪试验

日期：2026-10-01。目标是把目前的人源 ToR-ORd→Ca→Land 张力平台，和已完成的兔 MCT2024 机械反馈试验，接成有清楚边界的人体研究路线。本轮已完成第一版 ToR-ORd/SAC + Land 联合模型，把两条机械反馈路径独立开关并实际运行。

## 1. 文献和代码改变了“创新点”的位置

- Margara 等已经建立人心室 ToR-ORd/Land 双向耦合模型：电模型的游离 Ca 驱动 Land，而 Land 中结合于 troponin C 的 Ca 反过来更新游离 Ca。公开的 [CellContraction.jl](https://github.com/GSK-Biostatistics/CellContraction.jl) 实现已下载到 `Data/human_torland_reference`。因此，单纯宣称首次建立人 ToR-ORd/Land 双向模型不成立。来源：[Margara 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7848595/)、[作者代码](https://github.com/GSK-Biostatistics/CellContraction.jl)。本机尚无 Julia 运行时，此代码目前只审计了方程，**没有运行该实现**。
- Buonocunto 等已经在人心室 ToR-ORd 模型中分别加入非选择性、K 选择性和 Ca 选择性拉伸激活电流。其 [MATLAB 代码](https://github.com/melaniabuonoc/Electrophysiological-effects-of-stretch-activated-ion-channels) 已下载到 `Data/human_SAC_reference`，本文的可复核数值试验直接使用该 `model_SAC.m`。现有校准混合了人心房和鼠心室的拉伸电流数据，不能称为直接的人心室单细胞反馈增益测量。来源：[Buonocunto 2024](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP284439)。
- Timmermann 等此前已经在人和犬电生理背景中比较拉伸激活电流与肌丝 Ca 释放两条反馈，还分析了拉伸相位。因此本报告的双路径消融和相位变化是工程验证，不能作为独立生物学新发现。[Timmermann 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5716869/)。值得争取的新结果应是：**怎样用少量机械挑战和 AP/Ca/张力观测辨认具体细胞的反馈状态，并预测未见负载的反应**；这需要严格的反演、留出挑战与模型失配比较。

## 2. 一个解析可观测性结果：观测类型和挑战必须匹配

CellContraction.jl 的状态方程把 Land 参数 `beta_1` 写进 `ca50_eff = ca50 + beta_1 * min(0.2, lambda - 1)`，随后经 `dCa_TRPN` 进入游离 `dcai`。在 `lambda=1` 时，`beta_1` 从这条路径**严格消失**；再多基线 AP/Ca 数据也无法通过该方程辨认它。在 `lambda>1` 时，它才可能影响 Ca，进而影响膜电流和 AP。这是结构条件，实际能否估准还取决于噪声、参数补偿与刺激时机。代码位置：`Data/human_torland_reference/src/Model/Model.jl` 第 192、193、257 行。

同一实现中 `beta_0` 只在张力输出因子 `Lfac` 里出现，未进入状态方程。因此无论测多少 AP/Ca，**在这个实现里**都无法从 AP/Ca 单独识别 `beta_0`；必须增加张力观测或其他能看见此参数的观测。代码位置：`Data/human_torland_reference/src/Runner/Runner.jl` 第 17、33 行。这不是对所有 Land 实现的普遍断言。

这个分辨很重要：机械挑战确实能创建新信息，但不会使所有机械参数一起变得可辨识。

## 3. 实际运行：人心室 SAC 急性拉伸

用 Buonocunto 作者的 `model_SAC.m`、公开健康参数、1000 ms 周期和 `ode15s`（相对容差 1e-6，绝对容差 1e-8，最大步长 1 ms）运行。先从官方 ToR-ORd 初态预起搏，再在下一搏施加 10 ms 拉伸；每个实验与相同求解分段的 `lambda=1.00` 对照比较。读出是 1 ms 网格上的 `APD(V>-67 mV)`、Ca 峰值和拉伸后的最大逐时刻差值。协议代码：`human_sac_reference_pilot.m`、`human_sac_timing_grid.m`。

**预起搏复核。** 原先最后 10 搏的粗收敛标准在第 120 搏满足，但延长到论文使用的 500 搏后，对照 APD 由 251 变为 257 ms，Ca 峰值约从 0.4145 降到 0.4124 µM。第 491–500 搏 APD 范围 0 ms、Ca 峰值相对范围 7.0×10^-5。所以第 120 搏不是足够的长时稳态判据；以下主要读第 500 搏的拉伸**相对对照差值**。复核脚本：`human_sac_prepacing500_audit.m`、`score_human_sac_prepacing500.py`；数值：`human_sac_reference_pilot/prepace500_audit.json`。

| 500 搏后 10 ms 拉伸 | APD 相对对照 | 最大 Vm 相对变化 | 最大逐时刻 Ca 相对变化 | Ca 峰值变化 |
|---|---:|---:|---:|---:|
| 第 150 ms，λ=1.10 | +3 ms | +4.38 mV | +0.00258 µM | 0 |
| 第 600 ms，λ=1.10 | 0 ms | +8.12 mV | +0.00333 µM | 0 |

第 150 ms、`lambda=1.10` 情况下 APD 延长 3 ms；第 600 ms 尽管 APD 不变，拉伸后仍有明显的局部电压/Ca 波形差。两种情况都没有出现额外动作电位。500 搏图在 `human_sac_reference_pilot/prepace500_stretch_effect.png`。较早的 120 搏 3×3 时机/幅度网格也给出同方向变化，但应看作探索性时机扫描，而非独立验证；完整分数和图在 `timing_grid_score.json`、`stretch_timing_grid.png`。

**物理解释：** SAC 的影响依赖施加拉伸时细胞处于动作电位哪个相位。一次刺激可以明显改变局部膜电位，同时几乎不改变整搏 APD 或 Ca 峰值。所以若日后做人源校准，只保留 APD/Ca 峰值的摘要表会漏掉反馈信息；应尽可能保存事件对齐的 AP/Ca 全曲线和实际拉伸时序。这仍是公开模型内的结果，不是人体细胞实测效应大小。

## 4. 第一版闭环：SAC + 动态 Ca–TnC + Land 张力

已实现 `human_bidirectional_minimal.m` 和可复用 RHS `human_bidirectional_rhs.m`。联合状态是 43 个 ToR-ORd/SAC 状态加 7 个 Land 状态。Ca 由电模型进入 Land；Land 的动态 TRPN 结合速率被接回游离 Ca 方程；伸长同时改变 Land 长度依赖门控和 SAC 电流。两条反馈分别开关，做 2×2 pathway ablation。先从 500 搏 SAC 预起搏状态开始，再各跑 λ=1.00、1.05、1.10 一个持续等长搏动；所有反馈配置共享同一初态，比较各自与 λ=1 的变化。

这不是 Margara 参数化的原样复现：Land 状态方程采用本地公开 `Land2016.m` intact 默认值；Ca 反馈拓扑按 Margara 的动态 TnC 缓冲方式接回；机械输入是外部指定的肌节伸长比。因此目前是**给定长度输入的闭环细胞模型**，不包含由张力解出的力平衡/afterload 机械方程。SAC 关闭表示关闭随伸长变化的 SAC 支路；作者模型在 λ=1 仍保留基线 K 电流项。伸长依赖的有效膜电容因子也包含在作者的 SAC 电流实现里。

### 反馈通路消融结果

λ=1.10 持续伸长时：

| 动态 TnC-Ca feedback | SAC current | APD 改变量 | Ca 峰值改变量 | 解释 |
|---|---|---:|---:|---|
| 关 | 关 | 0 ms | 约 0 µM | 机械长度只改张力，电/Ca 保持不变 |
| 开 | 关 | +4 ms | −0.104 µM | stretch 改变 Ca50/TRPN 结合，重分配游离 Ca |
| 关 | 开 | +518 ms | +0.00028 µM | 在该参数与持续 10% 伸长下，SAC 强烈延迟复极 |
| 开 | 开 | +519 ms | −0.104 µM | 两条通路的读出特征基本可区分 |

APD 是本搏内 Vm 高于 −67 mV 的毫秒数。关闭两条反馈后，λ=1.00 对 λ=1.10 的最大差只有 0.000335 mV 和 0.0000075 µM，符合数值误差量级。500 搏闭环预起搏末 10 搏 APD 范围为 0 ms，Ca 峰相对范围 4.9×10^-5。更严容差重算后，λ=1.00/1.05/1.10 的 APD 分别为 258/305/777 ms；相对默认容差的 Vm 最大误差不超过 0.00056 mV，Ca 最大误差不超过 0.000004 µM。结果在 `human_bidirectional_output/ablation_audit.json` 与 `strict_solver_audit.csv`，图在 `human_bidirectional_output/bidirectional_ablation.png`。

这个结果证明程序中的双向路径确实闭合且每条路径都有独立作用。持续 10% 伸长下 SAC 造成的约 0.78 s APD 是非常强的模型反应，不能直接当作真实人心室效应预测：SAC 参数依赖混合物种数据，试验维持整个搏动的伸长，且本次只用一个健康参数组。后续解释应优先看短脉冲，不把 777 ms 当主生物学结论。

### 短脉冲和相位检查

随后施加 1 ms 上升、10 ms 保持、1 ms 释放的伸长脉冲，在 150、300、600 ms 三个相位扫描 λ=1.05、1.10；每个反馈配置均与它自己的 λ=1 对照比较。24 个挑战均未出现额外上冲。λ=1.10 时，150 ms 的 APD 改变量分别是：反馈全关约 0、仅 SAC +3.53 ms、仅 TnC +1.38 ms、两条都开 +5.28 ms。SAC 支路在 600 ms 相位不改 APD，但事件对齐 Vm 改变约 8.38 mV。动态 TnC 在 150 ms 相位使游离 Ca 瞬态最多下降约 0.077 µM，在 600 ms 相位最多下降约 0.009 µM；Ca 峰值均没有变化，因为峰值早于刺激出现。

对 λ=1.10、150/600 ms 的组合再用 `RelTol=1e-9, AbsTol=1e-11, MaxStep=0.1 ms` 重算；APD 分类与默认设置一致，关键逐时刻效应指标差异小于 6×10^-6 mV 和 1×10^-8 µM。表在 `human_bidirectional_output/pulse_audit.csv` 和 `pulse_strict_audit.csv`。这组试验支持相位依赖和路径分离的程序正确性，但只是一组模型内控制，不能当成人细胞观测。

### 一个尚待校准的基线差异

同样的作者 SAC 电模型在 500 搏静态 TnC 缓冲下，基线 Ca 峰约 0.4124 µM、APD 257 ms；接入 Land 动态 TnC 缓冲并再预起搏后，基线约 0.4963 µM、APD 258 ms。APD 接近，Ca 峰提高约 0.084 µM（约 20%）。[Margara 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7848595/)明确报告动态 TnC 耦合减少缓冲结合、提高游离 Ca，而 AP 变化很小，因此**偏移方向是有文献支持的**。本实现的 Land 参数并非 Margara 校准参数，具体 20% 幅度和张力波形仍需与原始参考生物标志物对照。故目前能确认反馈连接按方程工作且数值稳定，尚不能声称绝对 Ca transient 已经复现权威人模型。

## 5. 人体实测数据目前能做什么

本地 Dryad TTM `Data/Source+Data.zip` 有配对的人 hiPSC-CM 单层光学区域的 AP/Ca/收缩**特征**与药物扰动，能用于验证静态跨模态特征或药物响应。它没有全波形和指定的机械拉伸前后记录；收缩幅度也不等于测得的力。因此不能用它拟 SAC 增益，不能验证上面数毫伏的事件对齐反应，更不能把成人心室方程当成 hiPSC-CM 的已验证真值。完整实物审计见 `analysis/cardiac_three_download_audit.md`。

## 6. 下一步的最小整合实验与停止条件

1. 对照 Margara 原模型的参数和参考 AP/Ca/张力生物标志物，核对动态 TnC 造成的约 20% Ca 峰上升是否处于参考范围；该方向已有文献支持，幅度仍待确认。
2. 通过基线复核后，再对 SAC 与 beta1 做灵敏度/profile，判断能否凭 AP/Ca/张力将两条路径区分；随后扩展脉冲相位/幅度网格及不同参数组。
3. 真实人源机械干预数据若没有同样的拉伸定义、事件时序及 AP/Ca 测量，就只把联合模型称为**人源方程驱动的虚拟实验平台**。最终“个体化人心肌闭环孪生”必须通过独立人源机械挑战检验。兔 MCT2024 只提供试验设计和机械反馈候选的启发，不转移其数值 gain 或 period-2 阈值。

**当前结论：** 第一版人源方程驱动的双向电—Ca—收缩模型已运行，两条反馈路径通过消融得到分离，短脉冲结果对更严求解设置稳定。Ca 基线提高的方向吻合已有动态 TnC 耦合研究，幅度仍需参考模型定量验证。模型整合和拉伸相位效应已有明确先例；可能的新研究问题是有限机械挑战下的个体反馈可观测性和留出负载预测。
