## 仓库结构

- `kernal.py`：主脚本，定义 `VODKernelBinaryAnalyzer` 类，封装下列功能：
  - 生成在 $\{1,\dots,n_{\max}\}$ 上归一化的截断泊松先验；
  - 求解均衡响应率并计算无信息福利；
  - 通过对称搬运的 Jensen 检验评估局部凹性；
  - 计算解析核 $\mathcal{K}$ 并枚举相邻或全体状态对；
  - 穷举所有阈值划分，寻找使福利提升最大的二元信号；
  - 汇总结果、绘制 3×3 面板图，并导出阈值与核表格到 Excel。
- `敏感性分析.py`：复用 `VODKernelBinaryAnalyzer`，对 $(R,c,V,n_{\max})$ 参数网格进行灵敏度扫描，生成汇总统计、分 $\lambda$ 的长表、可选图像以及 Excel 报告。
- `proof_final (2).pdf`：伴随的理论推导与证明稿件（PDF）。
- `summary_kernel_binary_Kfinal.png`：运行 `kernal.py` 默认参数后得到的示例面板图。
- `results_kernel_binary_Kfinal_20250915_211259.xlsx` 与 `param_sensitivity_20250915_213728.xlsx`：对应主脚本与灵敏度脚本的示例输出，便于快速查看结果格式。
