# 原论文耦合肌节模型 × 人源电–钙模型：闭环仿真

## 状态

已运行一个**人源电–钙 × 原论文逐肌节力学的混合闭环**。代码直接调用 [Härtter 等 SarcomereModel](https://github.com/danihae/SarcomereModel) 的单肌节力–速度函数，随机版本保留其 Euler–Maruyama 更新和热/主动噪声形式。使用仓库提交 `f6feea542d9156f7985f1874d01491651de4435e`，许可证 MIT。人源电/钙使用本项目已移植的 ToR-ORd/SAC–Land 50 状态方程；20 根肌节各有位置和速度，总共 90 个状态。[原论文](https://elifesciences.org/articles/97321)

## 实现的闭环

```text
ToR-ORd 电活动 → Ca_i → Land TnC/薄丝开放比例
                           ↓
                  Härtter 单肌节主动力–速度律
                           ↓
             20 根肌节的位置 x_i、速度 v_i
                           ↓
         平均长度 λ 和速度 dλ/dt → SAC 电流
                           ↘ Land TnC–Ca 缓冲 → Ca_i
```

外部机械刺激沿原代码的负载耦合项进入，是 **250–350 ms 的 `p=-5` 无量纲预应变脉冲**。原代码中 `x_i` 增大代表缩短，转换为伸长比用 `λ_i=1−0.3x_i`；肌节激活用 `clip[(1−TmBlocked)/0.2,0,1]`。这两个 **0.3 / 0.2 是探索性接口比例尺**，目前没有同细胞数据可以校准。这里没有把原模型的无量纲力直接认作 kPa，也没有把 Land 主动张力再加进肌节加速度，因为那样会重复计算一次主动收缩。Land 张力在这一版是随状态计算的读数，**尚未做到两个机械模型之间的力平衡**。

## 结果与检验

所有消融共享同一人源预起搏状态和同一组初始肌节微小位置扰动。连续闭环在一次机械脉冲下的平均伸长比范围为 **0.968–1.035**。在脉冲及其恢复期，相比无机械脉冲，V 最大变化 **3.382 mV**、Ca 最大变化 **0.0163 µM**、λ 最大变化 **0.0396**。相同脉冲下关闭 SAC，V/Ca 分别相差 **3.860 mV / 0.0150 µM**。这些是此混合模型内部的因果消融，尚不是人体实验效应量。

更严格的路径检查：当“机械状态 → 电/Ca”反馈整体关闭时，同一机械脉冲仍使 λ 相差 **0.0397**，但 V/Ca 只相差 **0.000326 mV / 0.00000208 µM**（数值误差量级）。因此模型中机械脉冲影响电/Ca 的通路确实经过显式反馈接口。

| 数值审计 | 结果 |
|---|---:|
| 独立肌节随机更新 vs 原作者 `Model.integrate()` 的逐步最大差 | 1.6 × 10⁻¹⁴ |
| 连续闭环收紧求解容差与步长后，V 最大改变 | 0.00323 mV |
| 同上，Ca 最大改变 | 0.0000703 µM |
| 同上，平均 λ 最大改变 | 0.00000765 |
| 1 ms 随机分区求解在关噪声时 vs 连续闭环，V 最大差 | 0.302 mV |
| 同上，Ca 最大差 | 0.000371 µM |
| 20 ms 电/力交换窗的最大固定点 λ 更新 | 5.54 × 10⁻⁶ |

随机模型沿用原论文热噪声与主动噪声的系数 **2.2 / 9.0**，将原代码示例的 2 ms Euler–Maruyama 步长细化为 1 ms；同一随机种子下 20 根肌节出现不同步长度轨迹。**0.302 mV 的分区数值误差意味着本轮不能把随机版的亚毫伏电压差解释成可靠的生理效应。** 随机轨迹目前主要展示局部形变及闭环代码可运行性。

## 可运行文件

- `sarcomere_closed_loop.py`：直接调用作者肌节力–速度律的连续确定性闭环、消融、图与数组输出。
- `sarcomere_stochastic_cosim.py`：原随机项与 Euler–Maruyama 肌节更新、分区双向耦合。
- `validate_sarcomere_closed_loop.py`：原作者代码逐步等价性、求解器收敛和路径消融。
- `sarcomere_closed_loop_viewer/index.html`：离线交互页面，显示 20 根肌节的长度、膜、电/Ca 轨迹以及未配对的真实显微影像。由 `build_sarcomere_viewer.py` 生成数据。
- 数值结果：`sarcomere_closed_loop.json/.npz`、`sarcomere_stochastic_cosim.json/.npz`、`sarcomere_closed_loop_validation.json`。

在项目根目录运行：

```powershell
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\sarcomere_closed_loop.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\sarcomere_stochastic_cosim.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\validate_sarcomere_closed_loop.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\build_sarcomere_viewer.py
```

## 科学边界

1. 原论文的 `k_l` 和 `p` 是模型单位，尚无实验映射到基底刚度 kPa、细胞牵引力或外加应力。原模型的随机运动机制保留，但激活与长度接口为本项目新接的探索性假设。
2. 这是一条 20 根肌节的一维肌原纤维，不是经过影像分割的二维真实细胞形状；动画的细胞膜与离子通道空间位置均为示意。
3. 已有真实 ACTN2-Citrine 轨迹只观测形变，没有同一细胞同步 AP/Ca 与受控机械输入。它可用于轨迹分布对照，不能直接验证这里的电↔力闭环。
4. 人源 ToR-ORd/Land 与 Härtter 肌节模型来源不同。该混合架构的结构性正确性和数值可运行性已检验；参数、生理幅度和反事实预测仍需跨数据源标定/验证。

**当前可称：** 原论文肌节模型驱动的、可消融的人源力电闭环仿真原型。**当前不可称：** 已验证的人类细胞数字孪生或原论文模型的原样生理复现。
