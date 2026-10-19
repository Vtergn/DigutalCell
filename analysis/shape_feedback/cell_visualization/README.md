# 可视化心肌细胞：方程轨迹与真实肌节影像

打开 [`index.html`](index.html) 即可离线查看动画、四种反馈条件、膜电位/Ca/张力/长度曲线以及真实细胞的影像和追踪轨迹。`preview.png` 是 320 ms 时的静态预览。

GitHub 代码快照不包含从原始电影提取的 `real_5kpa_*.webp/png`。下载[原实验数据](https://doi.org/10.5281/zenodo.17564384)后按下述脚本重建；未重建时本地页面的真实影像区会缺素材，仿真曲线和示意动画仍可查看。

## 数据和可重复构建

- 仿真：项目已有的 51 状态 ToR-ORd/SAC–Land + 一维机械边界模型，`analysis/shape_feedback/embodied_length_pilot.py`。机械边界的刚度、阻尼和应力脉冲是探索性设置，并未经本页展示的真实细胞校准。
- 电流动画：对重新求解的状态轨迹，使用 `electrical_gating_partial.gating_intermediates` 计算 `INa`、`ICaL`、`IKr+IKs+IK1` 和 `INs`。发光强度按每种条件中的相应电流最大绝对值缩放，不是单通道开放概率，也不表示实际通道数。
- 真实影像：从 `Data/sarcomere_shape_pilot/5kPa_cell_01.zip` 提取 `summary_loi.png`，复制为 `real_5kpa_loi0_source.png`，图像内容未修改。使用 HTTP 字节范围从 [Zenodo 原始档案](https://zenodo.org/records/17564384) 的 **5.5 GB ZIP 中仅提取这一个 145 MB TIFF**，转换为 `real_5kpa_cell01_1p2-3p0s.webp`，不保留 TIFF，也不下载整包。展示窗口 1.2–3.0 s；转换时仅使用首帧固定亮度刻度和空间缩小，没有逐帧自动对比度。轨迹来自已提取的 `analysis/shape_feedback/sarcomere_tracks.npz`，细胞 01、LOI 0、相同的实际时间窗口。来源为 [Härtter 等，eLife，ACTN2-Citrine hiPSC-CM 研究](https://elifesciences.org/articles/97321)。
- 相关展示思想参考 [BAAIWorm 脑—体—环境仿真](https://www.nature.com/articles/s43588-024-00738-w)：动态模型与实验影像/轨迹并列呈现。本页现在有真实逐帧显微电影，但它与仿真不是同一刺激条件，不能宣称达到逐帧预测验证。

重新构建（约两分钟，项目虚拟环境需含 PyTorch、SciPy）：

```powershell
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\cell_visualization\build_viewer.py
& .\.venv_tcn\Scripts\python.exe analysis\shape_feedback\cell_visualization\build_real_movie.py
```

`viewer_data.js` 保存页面实际使用的轨迹；`validation.json` 保存与既有波形的逐点一致性核对。四种条件重新求解后，`V`、`Ca`、`λ` 与已有 `embodied_length_pilot.npz` 的最大绝对偏差均为 0（同一代码和求解环境）。
`real_movie_manifest.json` 记录原始 TIFF、提取帧号、播放间隔和强度显示规则。电影是 111 帧、920×180 像素，浏览器按 16 ms/帧循环播放；原追踪轨迹的采样间隔是 16.41 ms，因此动画比原记录快约 2.5%。`build_real_movie.py` 需要联网；已有 WebP 后打开网页不需要联网。

## 解释边界

渲染图中的双层膜、细胞核、肌原纤维、离子颗粒、通道坐标是视觉示意。长度变化跟随计算的 `λ(t)`，通道亮度跟随计算的电流，而**没有**空间分辨的膜、通道或胞内 Ca 场模型。不能把画面当作显微影像预测。

实验为 5 kPa **基底刚度**上的真实细胞；仿真为 12 kPa **外加应力**脉冲。两者不是相同控制量，时间轴与细胞身份也不配对。该窗口真实归一化长度最小值为 0.8794，仿真加载轨迹最小 `λ` 为 0.957471。页面展示这一差异，不把它计算成预测误差。实验电影未提供与该细胞同步的 AP/Ca/通道定位，因而目前只支持形态与收缩的定性对照，不支持完整力电闭环的真实数据验证。

## 下一层验证需要什么

若要达到类似数字线虫的逐帧对照，需要同一细胞、同一机械输入下同步获得细胞边界/肌节影像、膜电位或钙成像，以及实际力学边界条件；随后冻结模型参数，逐帧比较长度、收缩相位和电/钙响应。现有数据尚不满足这些配对条件。
