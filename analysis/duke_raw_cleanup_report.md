# Duke 原始文件空间清理

日期：2026-09-27。根据用户要求，清理 `Data/2022_pnas/2022_pnas` 下当前分析不直接读取的大型原始文件，保留完整 `Data/2022_pnas.zip` 作为可恢复备份。

## 删除范围与验证

- 823 个原始 `.asc` 测量文件，合计 86.01 GiB。
- 174 个文件名含 `rawdat`、`raw_dat`、`specdat` 或 `spec_dat` 的原始校准 CSV，合计 43.58 GiB。
- 共 997 个文件、129.59 GiB；具体相对路径、原始大小和 ZIP CRC-32 记录在 [清单](duke_raw_cleanup_manifest.json)。所有文件在删除前均匹配 ZIP 内同名成员及文件大小。
- 原始 ZIP 使用 Deflate64；Python 标准库不支持该压缩方法。因此使用本机 WinRAR 对整个 ZIP 做完整测试，退出码为 **0**，然后运行 [清理脚本](cleanup_duke_raw.ps1) 的 `-Execute` 模式。脚本在删除前验证根目录、扩展名、文件名和大小，并逐文件使用 PowerShell `Remove-Item -LiteralPath`，没有递归删除目录。

## 清理后核对

- 清单中的待删文件剩余 **0**。
- ZIP 保留，大小 **28.68 GiB**。
- 解压工作目录从 **141.11 GiB** 降为 **11.52 GiB**；其中 **371 个 `.feather`** 合计 10.99 GiB，另有 1,199 个 CSV 和图像、笔记本等。
- `bychannel_summary.csv`、`bychannel_approaches.csv`、`capchims_summary.csv`、`piezocd_summary.csv` 等关键汇总表保留。
- 重新运行 `analysis/inspect_duke.py` 成功；其读取的 100 个细胞汇总记录和 375,100 行压缩阶段记录仍可访问。

现有大多数波形分析直接读取 `.feather` 或汇总 CSV，可继续运行。`analysis/audit_alignment.py` 中与原始 `.asc` 对照的单个检查步骤，以及从头重做 AFM 校准/预处理，需要先从 ZIP 恢复相应原始成员。
