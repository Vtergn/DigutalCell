# 公开数据与参考模型下载

本仓库只保存本项目的代码、报告、小型结果表/图和 555 B 的名义预起搏初态 `analysis/torch_native_port/prepace_state_v5.mat`。`Data/`、`external/`、原始影像、仿真队列和模型权重不进入 Git。以下链接指向**原作者或官方数据仓库**；下载者应遵守各来源的许可和引用要求。无需为运行当前主线下载全部历史数据。

## 当前闭环主线

| 资源 | 官方入口 | 本地放置位置 / 用途 |
|---|---|---|
| Härtter 等肌节力学源码 | [SarcomereModel GitHub](https://github.com/danihae/SarcomereModel) | 克隆到 `external/SarcomereModel/`，使用提交 `f6feea542d9156f7985f1874d01491651de4435e`；90 状态闭环会直接导入其中的 `src/model`。 |
| 人 iPSC-CM 肌节运动影像与处理轨迹 | [Härtter 等 Zenodo 17564384](https://doi.org/10.5281/zenodo.17564384)；[论文](https://elifesciences.org/articles/97321) | 按需提取所需细胞，不必下载所有电影。本地分析使用的 5/15/85 kPa 样本包位于 `Data/sarcomere_shape_pilot/`；真实电影需按 `analysis/shape_feedback/cell_visualization/README.md` 重建。与仿真细胞**不配对**。 |
| ToR-ORd 电/钙参考实现 | [原作者 GitHub](https://github.com/jtmff/torord) | 若要核对原始 MATLAB/CellML 方程，可克隆到 `Data/torord_official/`；本项目 Python 移植源码已在 `analysis/torch_native_port/`。当前本地参考提交 `11e463da0f083c276c0057211e476f35cc2fc2c0`。 |
| Land 收缩参考实现 | [Physiome CellML 工作区 7fb](https://models.physiomeproject.org/workspace/7fb) | 下载 complete archive 到 `Data/7fb/`；含 `BG_crossbridge_TRPN.cellml` 和 `Land2016.m`。当前 Python 闭环使用本项目核对过的 Land 移植。 |

## 历史分析需要时再下载

| 研究分支 | 官方入口 | 本地路径与数据范围 |
|---|---|---|
| Duke AFM–Piezo 电流 | [Young 等 Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.gtht76hq5) | `Data/2022_pnas.zip` 与 `Data/2022_pnas/`。原站提供约 30.8 GB 压缩数据和 README；项目过去的大文件清理已使 ZIP 成为部分原始记录的恢复来源。仅复现这条分支时下载。 |
| ETH FluidFM/Flipper-TR 膜张力 | [ETH Research Collection 数据 DOI](https://doi.org/10.3929/ethz-b-000659903)；[论文与 Figure Source Data](https://www.nature.com/articles/s41592-024-02277-8) | `Data/StepwiseFlipperTrImaging/`、`Data/RealtimeFlipperTrImaging/`、`Data/CalciumImaging/` 及对应 ZIP。先下载论文 Figure 2/3 源表；完整影像体积较大。 |
| Piezo1 patch 张力门控 | [eLife 70988 Figure source data](https://elifesciences.org/articles/70988/figures) | Figure 4/5 的 XLSX 放 `Data/`；只用于独立物理先验，不与 Duke 按细胞 ID 拼接。 |
| PIEZO1 wedge / 脉冲间隔与 AFM 速度 | [Nature Communications 2026](https://www.nature.com/articles/s41467-026-76927-0) | 下载文章的 Source Data XLSX 到 `Data/`；本项目使用 `41467_2026_76927_MOESM4_ESM.xlsx` 中的指定工作表。 |
| PIEZO2–FLNB 扰动 | [Nature 2026 文章与 Source Data](https://www.nature.com/articles/s41586-026-10182-7)；[Zenodo 17644763](https://doi.org/10.5281/zenodo.17644763) | Figure/Extended Data XLSX 放 `Data/`。Zenodo 大型 raw ZIP 主要是成像资料；现有分析只需文章源表。 |
| Strocchi ToR-ORd–Land GSA | [Zenodo 7405335](https://zenodo.org/records/7405335)；[作者代码](https://github.com/MarinaStrocchi/Strocchi_etal_2023_GSA) | 只需 `ventricular_cell_model.tar.xz` 解到 `Data/ventricular_cell_model/`，代码可放 `Data/Strocchi_etal_2023_GSA-master/`；它是**仿真特征表**，不是实测 AP/Ca/T 波形。 |
| TTM 同步 AP/Ca/收缩特征 | [Dryad 67c92vm](https://datadryad.org/dataset/doi:10.5061/dryad.67c92vm) | `Source Data.zip` 放 `Data/`，解到 `Data/Source+Data/`。公开包是药物扰动的特征表，不是受控机械挑战全波形。 |
| 兔心肌闭环 MCT2024 | [作者 GitHub](https://github.com/UTokyo-FML/MCT2024)；[Zenodo 14269939](https://zenodo.org/records/14269939)；[实验示例 Dryad](https://datadryad.org/dataset/doi:10.25338/B8T935) | 源码可放 `Data/MCT2024-MCT_v1.0.0/`；实验示例放 `Data/doi_10_25338_B8T935__v20220607/`。兔模型与人模型分别分析。 |
| 独立人心肌 ORd–MechChem | [论文补充材料 Figshare](https://doi.org/10.6084/m9.figshare.12571949) | 原始源码可放 `external/mechchem_2020_code/`；无需下载约 3 GB 的整套预计算输出。详见本地 `PROVENANCE.md`。 |

## 从 GitHub 克隆后的最小复现

1. 建立 Python 3.13 虚拟环境并按根目录 `requirements-core.txt` 安装依赖；PyTorch wheel 应按机器平台选择。
2. 运行 `python scripts/project_check.py --source-only` 检查仓库文件。此检查不要求下载数据和第三方仓库。
3. 若需运行 90 状态逐肌节闭环，克隆上表的 SarcomereModel 到 `external/` 并固定所列提交。
4. 如需重算 51 状态两批虚拟细胞，在项目根目录运行：

   ```powershell
   python analysis/shape_feedback/run_embodied_cohort.py --seed 20261019 --out analysis/shape_feedback/embodied_cohort
   python analysis/shape_feedback/run_embodied_cohort.py --seed 20261020 --out analysis/shape_feedback/embodied_cohort_confirm
   python analysis/shape_feedback/blind_closed_loop_benchmark.py
   ```

   每批 12 细胞、每细胞 4 条协议，需要明显长于源码检查的计算时间。逐肌节协议先运行 `sarcomere_closed_loop.py` 生成名义轨迹，再运行 `sarcomere_protocol_suite.py`。完成后可运行 `python scripts/project_check.py` 检查生成工件。

上述流程只复现**同方程计算基准**。现有公开资料没有一套与当前虚拟细胞逐个配对的同步 AP、Ca、受控机械输入、力和形变记录，不能据此报告真实单细胞预测精度。
