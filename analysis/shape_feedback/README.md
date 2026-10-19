# 细胞形变与力电闭环：2026-10-18 可复现实验记录

2026-10-19 新增时变干预接口与封存协议的四模态盲测；见
[`action_conditioned_virtual_cell_report.md`](action_conditioned_virtual_cell_report.md)。

## 问题与数据

目标是检验能否沿用 BAAIWorm 的“生理状态—身体—环境—反馈”思路，让心肌细胞的形变由内部收缩力和外部负载**共同产生**，并用真实显微轨迹检验学习形变动力学的可行性。BAAIWorm 原论文：https://doi.org/10.1038/s43588-024-00738-w 。

真实形变数据来自 Härtter 等 eLife 文章（https://elifesciences.org/articles/97321）公开的 Zenodo 记录 https://doi.org/10.5281/zenodo.17564384 。本次只从大 ZIP 中按需提取六个单细胞的已处理 LOI 轨迹：5、15、85 kPa 各两颗。六个 ZIP 存放于 `Data/sarcomere_shape_pilot/`，对应脚本为 `remote_zip_select.py`、`extract_sarcomere_tracks.py`。没有下载 TIFF 原始电影或约 11 GB 的汇总 pkl。六个 ZIP 合计约 571 MB。

每颗细胞有多条 LOI，这些是**同一细胞的相关观测**，不能当作独立细胞。帧间隔约 16.4 ms。5 kPa 样本为 day 23/27，15 kPa 为 day 28/32，85 kPa 为 day 19；成熟日龄和基底刚度明显混杂，因此当前样本不能识别独立的刚度效应。该实验没有同一细胞同步 AP、Ca 和 LOI，也没有急性、受控 stretch pulse。

## 已建立的方程闭环

`embodied_length_pilot.py` 将既有 50 状态 ToR-ORd/SAC–Land 系统增加一个内生长度状态 λ：

\[
\eta\dot\lambda=P_0+P_{\rm ext}(t)-K(\lambda-1)-T_{\rm Land}(x,\lambda),
\qquad
\dot x=f_{\rm human}(x,\lambda,\dot\lambda).
\]

这里张力由 AP→Ca→Land 产生；求得的 λ 又同时进入 Land 和 SAC/TnC 反馈。外界输入是 stress pulse，而不是预先指定的长度轨迹。因此这是一维力学自由度的**实际闭环求解**。K=220 kPa/stretch、η=5500 kPa·ms/stretch 和 12 kPa pulse 是探索性工程参数，**尚未由真实细胞辨识**。λ<1 时电学 SAC 使用 max(λ,1) 以避免既有分数幂 SAC 在压缩区未定义；这一截断也是明确的模型假设。

在 250–350 ms 施加 12 kPa 外载，求解出的 λ 最大到 1.0451，收缩时最小到 0.9575。相同外载下开/关 SAC，电压轨迹最大差异 4.55 mV，Ca 最大差异 0.0170 µM。这个差异是**模型内部的通路消融效应**，不能解释为已在人细胞测到相同效应。关闭 TnC 的组合实验还会改变基础状态，不能作为单一 TnC 因果消融结论。

`validate_embodied_length.py` 给出数值检查：严格/默认 BDF 解的电压 MAE 0.000278 mV、Ca MAE 1.44×10⁻⁶ µM、λ MAE 8.53×10⁻⁷。该检验支持数值稳定性；**不构成生物学参数验证**。

`render_embodied_length.py` 依据求出的 λ 制作 `embodied_length_schematic.gif`。椭圆轮廓及面积守恒宽度规则只是示意渲染，尚未学习或求解真实二维细胞轮廓。

## 真实形变数据的神经网络去留测试

`fit_shape_dynamics.py` 用最近 12 帧平均肌节长度和基底刚度，预测未来 10 帧（约 164 ms）。5 与 85 kPa 的四颗细胞训练，两颗 15 kPa 细胞整体留出；细胞内 LOI 不能跨训练/测试分配。与 persistence 和 ridge 比较。对每颗留出细胞的额外校准只使用它前 300 帧（约 4.9 s），之后所有预测窗口仍封存。旧的三细胞先导结果保存在 `shape_dynamics_three_cell_pilot.json`。

| 测试（15 kPa 两细胞，后续 1716 窗口） | 10 帧平均绝对误差，归一化肌节长度 | 第 10 帧误差 |
|---|---:|---:|
| 当前长度保持不变 | 0.00419 | **0.00725** |
| Ridge | **0.00386** | 0.00743 |
| MLP，不做个体校准 | 0.00939 | 0.01698 |
| MLP，前 300 帧校准增益 | 0.00424 | 0.00728 |

MLP 在训练细胞的后段验证较好（0.00437，persistence 为 0.00574），但换细胞和刚度条件就退化。校准让 MLP 测试误差近乎减半，仍未胜过 ridge，也只与 persistence 接近。两个留出细胞分别为：MLP 校准后 0.00553/0.00323；对应 ridge 0.00500/0.00296。窗口数不是独立样本数；**独立测试细胞只有两颗**。这项试验支持“需要个体校准”的动机，不能支持“神经网络已还原细胞形变”或“已学到刚度—形变规律”。

固定同两颗测试细胞进行小型训练规模对照：只用 2 颗训练细胞时，原 MLP / 校准 MLP / Ridge 的误差分别是 0.01207 / 0.00463 / 0.00463；增至 4 颗训练细胞后为 0.00939 / 0.00424 / 0.00386。新增细胞对三者都有一定帮助，原 MLP 约降 22%；但样本仍太少，不能据此估计可靠的样本量曲线，也没有证明 MLP 最终会超过简单基线。对应文件为 `shape_dynamics_pilot_two_train.json` 与 `shape_dynamics_pilot.json`。

更重要的是，该网络输入是既往形变轨迹，而不是同步 AP/Ca/T；它目前没有嵌入上述 51 状态反馈方程。把它接进去会制造未经验证的 Ca→形变映射。当前成果应称为“一维方程闭环 + 真实肌节轨迹学习可行性审计”，不能称为 BAAIWorm 级别的多尺度、神经网络驱动细胞体模型。

## 可复现方式

使用项目现有 `.venv_tcn/Scripts/python.exe`。数据提取后：

```powershell
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\extract_sarcomere_tracks.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\fit_shape_dynamics.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\embodied_length_pilot.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\validate_embodied_length.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\render_embodied_length.py
```

主输出：`sarcomere_tracks.png`、`shape_dynamics_pilot.png/json`、`embodied_length_pilot.png/json`、`embodied_length_validation.json`、`embodied_length_schematic.gif`。

## 下一道证据门槛

在不补全受控加载、AP/Ca 与肌节/轮廓同步观测的前提下，可继续做更大量的**形变时序**训练与留细胞验证；它仍不能建立真实的电→形变→电闭环。若要走到类似 Digital Worm 的可验证平台，需要同一细胞的时间同步 AP/Ca、形变（最好二维轮廓或位移场）、外载及其边界条件。随后才有理由把神经网络限制在未知的 Ca/T→局部形变或黏弹性通量，并用未见机械刺激做盲预测。

逐肌节空间信息与三阶统计量的留刚度条件去留测试见 `spatial_information_gate_report.md`；目前不支持将这些轨迹解释为已发现的力电联合流形。

在**给定未来整条 LOI 平均长度**的条件下，逐肌节形变解码的留刚度检验见 `spatial_decoder_gate_report.md`。局部松弛与相邻项降低预测误差，但相邻次序相对于随机拓扑的独立贡献很小；这是运动层的条件预测检验，并非实验确认的电↔力闭环。

在此之后已接入 [Härtter SarcomereModel](https://github.com/danihae/SarcomereModel) 原始肌节力–速度与随机更新代码，建立 Ca/TnC→逐肌节运动→平均 λ→SAC/TnC→电/Ca 的混合闭环；数值与消融见 `sarcomere_closed_loop_report.md`。可离线查看 `sarcomere_closed_loop_viewer/index.html`。此接口的力、长度和激活比例尺仍为探索性假设，不能把现有未配对实验影像视为闭环验证。
