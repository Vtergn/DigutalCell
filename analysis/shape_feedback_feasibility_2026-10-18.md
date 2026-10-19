# 细胞形变网络与具身式力电闭环：数据可行性审计

日期：2026-10-18。目的：核对能否借鉴 BAAIWorm 的“内部状态→身体形变→环境作用→感觉反馈”结构，建立会改变自身形状、且形变能反作用于 AP/Ca 的虚拟心肌细胞。

## 结论

**计算原型可做；真实人源、影像约束的闭环孪生尚未由现有数据支持。** 最合适的起点是人源心肌细胞，而不是把 Piezo、红细胞与心肌数据拼成一个虚构个体。第一版应学习**肌节长度/细胞轮廓或位移场的动力学**，并让形变状态真实进入 SAC 与 TnC-Ca 通路。渲染必须从被求解的形变状态生成；单纯把动画宽度随预设 λ 改变，不构成形变模型。

BAAIWorm 的参照是其架构和多层验证：生物物理神经网络驱动有限元身体，身体与环境相互作用，再把感觉信号送回神经网络；它不是“视频生成器 + 电学预测器”。[原论文](https://www.nature.com/articles/s43588-024-00738-w)

## 本地数据逐项核对

| 本地资源 | 已有的真实/模拟观测 | 对形变闭环的用途 | 不足 |
| --- | --- | --- | --- |
| `analysis/torch_native_port/native_bidirectional.py` | 经 MATLAB 对照的 50 状态 ToR-ORd/SAC-Land RHS；输出 V、Ca、张力；λ 输入 SAC/Land | 保留电、Ca、肌丝和反馈核心 | λ 是外加量；没有从力平衡求出自由形变，也没有空间轮廓 |
| `analysis/virtual_cell_decoder` | 47 个有效合成细胞，35/12 细胞划分；网络预测 V/Ca/T | 快速 surrogate 与可视化脚手架 | GIF 椭圆宽度直接取**输入 λ**，没有影像形变真值；不是闭环形变预测 |
| `Data/ttm_public` | 公开 hiPSC-CM 区域级 AP/Ca/收缩特征；药物前后配对 | 三模态时间尺度/终点合理性检查 | 无公开视频、逐帧轮廓及受控长度挑战；非单细胞张力 |
| `Data/Dataset_supporting_the_publication` | HL60 流动形变指标 + 多轴阻抗及 CytoD 等扰动 | 静态形状/电阻抗表型检查 | 阻抗不是膜 AP；无同细胞连续形变→离子反馈 |
| `Data/2022_pnas` | HEK/Piezo 的 AFM 力、压入与电流 | 机械刺激→电流通路参考 | 无随时间的细胞轮廓；不能用它训练心肌形变 |
| `Data/CalciumImaging`、`RealtimeFlipperTrImaging` | ETH 系统的 Ca/膜张力影像 | 力学局部性和 Ca 响应的辅助约束 | 不同细胞体系；无心肌 AP、肌节或心肌收缩输出 |
| 红细胞 EDS 分支 | 电场下轮廓变形 | 训练被动细胞形变/电场力学可行性 | 外部 AC 极化不等于细胞膜兴奋；不提供心肌反馈通路 |

核对依据：`analysis/torch_native_port/PORT_STATUS.md`、`analysis/virtual_cell_decoder/README.md`、`analysis/train_conditioned_behavior_decoder.py`、`analysis/ttm_public_data_audit.json`、Southampton 的 `Readme.txt`。现有渲染代码明确把 `mem.width` 设为 `1.5*(1+3*(lam[p,q]-1))`；因此不能把 GIF 当作网络学会细胞形变的证据。

## 公开数据候选及适配程度

1. **Härtter 等，eLife 2026：人 iPSC-CM 肌节动态。** 六种基底刚度，公开每组代表性原始显微电影和对应逐肌节轨迹。最适合训练/检验“边界刚度、运动历史→肌节运动”的*机械观测层*，尤其动态异质性和负载依赖。文章没有报告与这些电影同细胞同步的膜电位、Ca 和受控急性长度挑战，所以不能直接监督训练 `Ca→形变`。先审计处理后的轨迹、细胞 ID、时间戳和下载体量，不先拉全部影像。[论文](https://elifesciences.org/articles/97321)；[数据](https://doi.org/10.5281/zenodo.17564384)。
2. **CONTRAX，Nature Communications 2024：单个人 iPSC-CM 的轮廓、位移和牵引力。** 2753 个符合 QC 的单细胞，多种基底刚度与药物条件；代码、source data 公开。适合形状/位移观测算子和外部机械验证。采集电刺激，但未同步记录这些细胞的 AP/Ca；其原始影像归档极大（JHU 提供的影像包约 582 GB），应先从 OSF/source data 与少量样本着手。[论文](https://www.nature.com/articles/s41467-024-49755-3)；[代码](https://github.com/MicrosystemsLab/ContraX)；[影像归档](https://archive.data.jhu.edu/file.xhtml?datasetVersionId=806&fileId=11916)。
3. **TTM，Nature Communications 2019：AP/Ca/收缩同步光学观测。** 原实验确实三通道逐帧同步采集，适合制定共同观测定义；当前本地公开文件主要是特征，缺能用于影像网络训练的完整逐细胞电影，也无受控机械挑战。[原论文](https://www.nature.com/articles/s41467-019-12354-8)。
4. **openCARP single-cell electromechanics。** 开源参考实现有强耦合 MEF，以及 `tomek20land17_dynsarc` 的动态肌节长度，适合核对“内生 λ”接口和力学边界。它是已发表模拟器参考，不是新的实验影像真值。[官方例子](https://opencarp.org/documentation/examples/01_ep_single_cell/06_em_coupling)。
5. **Peussa 等 Zenodo：基底光致形变 + Ca/明场。** 机械扰动和 Ca 影像同体系，是独立机械→生化验证的候选；为 MDCK 上皮细胞、采样 1.23 s/帧，无法直接约束毫秒级心肌 AP 反馈。[数据](https://zenodo.org/records/8215150)。

## 最小可证伪的模型

先求解真正的机械状态，而不是指定形变轨迹：

```
pacing → V/Ca → Land active tension → mechanical force balance
             ↑                              ↓
      SAC and TnC-Ca feedback ← λ(t), λdot(t), shape q(t)
```

例如第一版取一维内生肌节长度 λ 和黏弹性环境：`η λdot = F_external(t) - F_passive(λ) - F_active(Ca, myofilament_state, λ)`，再将 λ、λdot 送回现有电/Ca/肌丝 RHS。这个符号约定与量纲只是**候选接口**，必须由现有 Land 方程的张力定义、细胞横截面和边界条件推导后才能实际接入。第二版让小网络只学习 `Ca、λ、λdot、刚度、历史 → 机械残差/低维位移`，其输出通过力平衡和 SAC/TnC-Ca 反馈到下一时刻。公开运动轨迹可以约束网络的**边缘运动统计和负载依赖**；由于缺同细胞 Ca 配对数据，`Ca→形变` 部分只能先由机制模型/合成数据训练，不能称为实验监督训练。严禁让网络直接任意改写整个 `dV/dt`。

渲染用数值计算得到的轮廓节点/位移场；验证目标是轨迹、轮廓和张力，而不是视觉逼真程度。

## 按顺序执行的 go/no-go

1. **数据门槛：**选一个公开形变数据集，确认至少有可追踪的逐细胞/肌节时序、已知帧率、负载条件和细胞级 split。若仅有静态汇总，不训练动态形变网络。
2. **力学门槛：**从现有 RHS 接内生长度/力平衡；在无电反馈时复现已发表的长度-张力、缩短速度/相位关系；核对被动、主动功与能量耗散。若力学不合理，不进入 NN。
3. **网络门槛：**用未见细胞和未见刚度/负载测试长时自由 rollout。比较物理力平衡、纯数据序列模型与物理约束残差；同时报形变误差和受力误差。仅一步预测准确不算通过。
4. **闭环门槛：**打开/关闭 SAC、TnC-Ca、力学 NN 三组消融。保持同一起搏和外力，形变必须先变化，随后 AP/Ca 按通路因果改变；换负载/起搏后仍稳定。比较自由形变与外加 λ 两种模拟，明确新闭环带来的差别。
5. **现实门槛：**公开影像可分别验证机械层；公开 AP/Ca 数据可分别验证电层。**没有同一细胞的机械输入、形变和 AP/Ca 配对序列，就不能宣称真实人体闭环数字孪生已验证。**

建议先完成 1D 肌节/长度版，再扩到 2D 轮廓/位移网络；空间网络需要显著更多影像、边界条件与受力信息。新意应放在“影像约束的形变状态是否改善未见机械挑战下的双向电-机械预测”，而不是仅仅把现有输出画成动图。
