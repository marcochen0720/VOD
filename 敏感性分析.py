# -*- coding: utf-8 -*-
"""
Parameter Sensitivity for VOD + Binary Threshold Signals
- Reuses your class: VODKernelBinaryAnalyzer (must be defined/imported above).
- Produces:
    * summary DataFrame over (R,c,V,n_max) combos
    * per-(combo,lambda) long table
    * optional plots
    * Excel export
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from kernal import VODKernelBinaryAnalyzer  # 确保你的类定义在 kernel.py 或同一文件中
# ------------------------------------------------------------
# 1) 核心：对单个参数组合做一次 λ 扫描并汇总
# ------------------------------------------------------------

def sweep_one_combo(R, c, V, n_max, lam_values, analyzer_cls):
    """
    对单个 (R,c,V,n_max) 组合做 λ 扫描。返回：
      - summary_row: 该组合的汇总指标（字典）
      - long_table: 逐 λ 的细表（DataFrame）
    """
    analyzer = analyzer_cls(n_max=n_max, R=R, c=c, V=V)

    rows = []
    for lam in lam_values:
        prior = analyzer.prior_poisson_trunc(lam)

        # 局部凹性（核K的Jensen检验）
        loc = analyzer.local_concavity_kernel(prior, eps=1e-3, n_pairs=80)

        # 二元阈值扫描（全阈值，取最优）
        best, table = analyzer.best_binary_improvement(prior)

        rows.append(dict(
            R=R, c=c, V=V, n_max=n_max, lam=float(lam),
            c_over_R=c/R,
            welfare_prior=loc['v0'],
            p_prior=loc['p0'],
            locally_concave=bool(loc['is_locally_concave']),
            max_jensen_gap=float(loc['max_gap']),
            best_threshold=int(best['threshold']),
            tau_L=float(best['tau_L']), tau_H=float(best['tau_H']),
            vL=float(best['vL']) if not np.isnan(best['vL']) else np.nan,
            vH=float(best['vH']) if not np.isnan(best['vH']) else np.nan,
            EV=float(best['EV']),
            improvement=float(best['improvement']),
            improvement_pct=float(best['improvement_pct']),
            pL=float(best['pL']) if not np.isnan(best['pL']) else np.nan,
            pH=float(best['pH']) if not np.isnan(best['pH']) else np.nan
        ))

    long_table = pd.DataFrame(rows)

    # 汇总指标
    # 峰值改进（%）
    idx_max = long_table['improvement_pct'].idxmax()
    lam_at_max = float(long_table.loc[idx_max, 'lam'])
    max_impr_pct = float(long_table.loc[idx_max, 'improvement_pct'])
    max_impr_abs = float(long_table.loc[idx_max, 'improvement'])
    # 平均改进（%）
    avg_impr_pct = float(long_table['improvement_pct'].mean())
    # 产生正改进的 λ 比例
    share_positive = float((long_table['improvement'] > 1e-12).mean())
    # 角点比例
    share_corner = float((long_table['p_prior'] > 0.999).mean())
    # 非凹比例（Jensen 违反）
    share_nonconcave = float((~long_table['locally_concave']).mean())

    summary_row = dict(
        R=R, c=c, V=V, n_max=n_max, c_over_R=c/R,
        lam_min=float(np.min(lam_values)), lam_max=float(np.max(lam_values)),
        max_improvement_pct=max_impr_pct,
        max_improvement_abs=max_impr_abs,
        lam_at_max=lam_at_max,
        avg_improvement_pct=avg_impr_pct,
        share_positive_improvement=share_positive,
        share_corner=share_corner,
        share_nonconcave=share_nonconcave
    )

    return summary_row, long_table


# ------------------------------------------------------------
# 2) 扫描一组参数网格
# ------------------------------------------------------------

def run_sensitivity_grid(R_list, c_list, V_list, nmax_list, lam_values, analyzer_cls,
                         outdir="results_param_sensitivity", do_plots=True, save_excel=True):
    os.makedirs(outdir, exist_ok=True)

    all_summary = []
    all_long_tables = []

    # 主循环：网格扫描
    for R in R_list:
        for c in c_list:
            for V in V_list:
                for nmax in nmax_list:
                    srow, ltab = sweep_one_combo(R, c, V, nmax, lam_values, analyzer_cls)
                    all_summary.append(srow)
                    all_long_tables.append(ltab)

                    # 可选：画该组合的 improvement% vs λ
                    if do_plots:
                        fig, ax = plt.subplots(figsize=(6.5, 4.0))
                        ax.plot(ltab['lam'], ltab['improvement_pct'], 'o-', lw=1.5)
                        ax.axhline(0, color='k', ls='--', alpha=0.4)
                        ax.set_xlabel(r'$\lambda$')
                        ax.set_ylabel('improvement (%)')
                        ax.set_title(f"Improvement% vs λ | R={R}, c={c}, V={V}, n_max={nmax}, c/R={c/R:.3f}")
                        ax.grid(alpha=0.3)
                        fpath = os.path.join(outdir, f"imprpct_vs_lambda_R{R}_c{c}_V{V}_n{nmax}.png".replace('.', '_'))
                        fig.savefig(fpath, dpi=200, bbox_inches='tight')
                        plt.close(fig)

    summary_df = pd.DataFrame(all_summary)
    long_df = pd.concat(all_long_tables, ignore_index=True)

    # 可选：固定 (V,n_max)，展示 max_improvement_pct 随 c/R 的曲线（若 c,R 扫描>1个）
    if do_plots and (len(V_list) == 1 and len(nmax_list) == 1) and (len(R_list) * len(c_list) > 1):
        V0, n0 = V_list[0], nmax_list[0]
        sub = summary_df[(summary_df['V'] == V0) & (summary_df['n_max'] == n0)].copy()
        sub = sub.sort_values('c_over_R')
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(sub['c_over_R'], sub['max_improvement_pct'], 'o-', lw=1.8)
        ax.set_xlabel('c/R')
        ax.set_ylabel('max improvement (%)')
        ax.set_title(f'Max improvement% vs c/R | V={V0}, n_max={n0}')
        ax.grid(alpha=0.3)
        fpath = os.path.join(outdir, f"max_imprpct_vs_c_over_R_V{V0}_n{n0}.png".replace('.', '_'))
        fig.savefig(fpath, dpi=200, bbox_inches='tight')
        plt.close(fig)

    # 导出 Excel
    xlsx_path = None
    if save_excel:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = os.path.join(outdir, f"param_sensitivity_{ts}.xlsx")
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            long_df.to_excel(writer, sheet_name='ByLambda', index=False)
    return summary_df, long_df, outdir, xlsx_path


# ------------------------------------------------------------
# 3) 使用示例（你可以根据需要改网格）
#    —— 确保你已经定义了 VODKernelBinaryAnalyzer 类
# ------------------------------------------------------------

if __name__ == "__main__":
    # ===== 你的类：请确保已定义在同一文件上方 =====
    from __main__ import VODKernelBinaryAnalyzer  # 如果放在另一个文件，请改成相应的 import

    # λ 扫描范围（整数点）
    lam_values = np.arange(1, 21, 1.0)

    # 参数网格（示例：两三档就能看趋势；需要更细可自行加密）
    R_list = [8.0, 10.0, 12.0]
    c_list = [0.8, 1.0, 1.2]
    V_list = [30.0, 50.0]      # 降低 V 会抬高“百分比改进”
    nmax_list = [30]           # 如需看尾部影响，可试 [30, 50]

    summary, bylambda, outdir, xlsx = run_sensitivity_grid(
        R_list=R_list, c_list=c_list, V_list=V_list, nmax_list=nmax_list,
        lam_values=lam_values,
        analyzer_cls=VODKernelBinaryAnalyzer,
        outdir="results_param_sensitivity",
        do_plots=True, save_excel=True
    )

    print("\n=== 参数敏感性：汇总（Top 10 按峰值改进%） ===")
    print(summary.sort_values('max_improvement_pct', ascending=False).head(10).to_string(index=False))
    if xlsx:
        print(f"\nExcel 导出：{os.path.abspath(xlsx)}")
    print(f"图像目录：{os.path.abspath(outdir)}")