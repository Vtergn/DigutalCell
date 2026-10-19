# 工作区清理记录

## 2026-10-19

清理候选限定为 `analysis/` 下的 `__pycache__`、离线展示预览产生的 `chrome_profile`、`tmp/papers` 与 `tmp/pdfs`。这些都是可由源码或原 PDF 再生的缓存，不含下载数据和论文数值输出。删除前通过 `scripts/clean_generated.ps1` 校验目标的解析后绝对路径位于本项目根目录内部；本次删除 **8 个目录、433 个文件、11.11 MB**。脚本复查应返回 0 个候选目录。

特意保留：`Data/2022_pnas.zip`（此前大文件清理的恢复备份）、ETH ZIP、`analysis/` 各模型 `.mat/.npz`、原始实验影像、外部代码及 `.venv_tcn`。这些文件即使占空间也不是可随意认定为废弃的缓存。

