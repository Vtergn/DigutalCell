# 数据来源与存储原则

## 目录职责

- `Data/`：用户下载的论文原始数据、压缩档、第三方模型。默认**只读**。本项目生成物放在 `analysis/` 下，避免覆盖来源数据。
- `external/SarcomereModel/`：Härtter 等的第三方 MIT 代码，已用于 90 状态混合闭环；原仓库提交 `f6feea542d9156f7985f1874d01491651de4435e`。本项目新写的耦合代码在 `analysis/shape_feedback/`。
- `analysis/`：脚本、报告和带条件标记的生成产物。报告中的数值应指向对应 `.json`/`.npz`/`.mat`，图只是展示。

## 当前空间盘点（2026-10-19）

| 位置 | 约占用 | 保留原因 |
|---|---:|---|
| `Data/` | 66 GB | 下载档与提取的数据；含 Duke、ETH、原论文模型等 |
| `Data/2022_pnas.zip` | 28.68 GiB | 曾从展开目录删除 129.59 GiB 原始记录；该档案是恢复备份，见 `analysis/duke_raw_cleanup_report.md` |
| `analysis/` | 3.04 GB（清理前） | 多个已完成实验的数值记录；最大的为 MCT 与 MechChem 方程输出 |
| `.venv_tcn/` | 0.64 GB | 本机环境；可由依赖重建，Git 忽略 |

ZIP 与已提取目录可能看似重复，但 ZIP 有时是清理原始测量后的唯一恢复路径；没有逐文件核对前**不删除**。尤其 `2022_pnas.zip` 需按既有清理报告保留。`analysis/mechchem_cross_model/smoke_output.mat` 被 `check_first_second_beat.m` 读取；其他大 `.mat` 也可能支撑已发表图表或数值审计，未作批量删除。

本轮只清理确定可再生的 Python 字节码、一次浏览器预览的 Chrome profile，以及 `tmp/` 中从原 PDF 派生的临时文本/图片。清理脚本默认预览：`scripts/clean_generated.ps1`；加 `-Execute` 才实际删除。每次删除应记录时间、目标和验证方式。

