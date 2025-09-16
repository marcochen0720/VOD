# -*- coding: utf-8 -*-
"""
VOD：核 K 的凹性与二元信号的信息改进（最终版，含 K 判别与 tau 防呆）
- 先验：截断泊松（θ>=1），在 {1,...,n_max} 上归一化
- 均衡：Σ μ(θ)(1-p)^(θ-1) = c/R；边界 μ(1)≥c/R => p*=1
- 局部非凹性（Jensen）：对称搬运 ε 质量做 Jensen 检验
- 解析判别核 K：K = (B/Gp) Δϕ' - 2 Δv'，相邻/全对扫描
- 全局二元信号：阈值扫描，严格计算 τ_L, τ_H，断言 τ_L+τ_H=1
- 输出：图 + Excel（含每个 λ 的阈值扫描表与核表）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.special import factorial
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# --------- 中文字体 ---------
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False


class VODKernelBinaryAnalyzer:
    def __init__(self, n_max=30, R=10.0, c=1.0, V=50.0):
        self.n_max = int(n_max)
        self.R = float(R)
        self.c = float(c)
        self.V = float(V)
        self.states = np.arange(1, self.n_max + 1, dtype=int)

    # ------------------ 先验（截断泊松并归一化） ------------------
    def prior_poisson_trunc(self, lam: float) -> np.ndarray:
        # pmf for θ=1..n_max
        raw = np.array([lam**k * np.exp(-lam) / factorial(k) for k in self.states], dtype=float)
        # 条件化：去除 θ=0 的概率，再在 {1..n_max} 上归一化
        denom = 1.0 - np.exp(-lam)
        raw = raw / denom
        prior = raw / raw.sum()
        # 防呆
        prior = np.maximum(prior, 0.0)
        prior = prior / prior.sum()
        return prior

    # ------------------ 均衡 p(μ) ------------------
    def equilibrium_p(self, posterior: np.ndarray) -> float:
        target = self.c / self.R
        mu1 = float(posterior[0])  # μ(1)
        # 边界：F(1-)=μ(1)-c/R
        if mu1 >= target - 1e-12:
            return 1.0

        def F(p):
            if p <= 0.0 or p >= 1.0:
                return np.inf
            lhs = np.sum(posterior * (1.0 - p) ** (self.states - 1))
            return lhs - target

        try:
            return brentq(F, 1e-8, 1.0 - 1e-8)
        except ValueError:
            # 理论上不会触发；作为兜底
            def sq(p):
                if p <= 0.0 or p >= 1.0:
                    return 1e9
                return (F(p)) ** 2
            res = minimize_scalar(sq, bounds=(1e-6, 1.0 - 1e-6), method='bounded')
            if res.success:
                return float(np.clip(res.x, 1e-6, 1.0 - 1e-6))
            # 最后兜底：返回一个保守的内点
            return 0.5

    # ------------------ 福利 v(μ) ------------------
    def welfare(self, posterior: np.ndarray):
        p = self.equilibrium_p(posterior)
        theta = self.states.astype(float)
        # 生命成功概率
        success = 1.0 - (1.0 - p) ** theta
        # 唯一响应期望
        unique = theta * p * (1.0 - p) ** (theta - 1.0)
        # 总响应
        total = theta * p
        v = float(np.dot(posterior, self.V * success + self.R * unique - self.c * total))
        return v, p

    # ------------------ Jensen 局部凹性（对称搬运） ------------------
    def local_concavity_kernel(self, prior: np.ndarray, eps=1e-3, n_pairs=60, rng_seed=12345):
        v0, p0 = self.welfare(prior)
        n = len(prior)

        # 候选对（两边都有余量 ≥ eps）
        candidates = [(i, j) for i in range(n) for j in range(n)
                      if i != j and prior[i] >= eps and prior[j] >= eps]
        if not candidates:
            return dict(is_locally_concave=True, n_pairs_tested=0, n_viol=0,
                        max_gap=0.0, v0=v0, p0=p0)

        rng = np.random.default_rng(rng_seed)
        idxs = rng.choice(len(candidates), size=min(n_pairs, len(candidates)), replace=False)

        viol_gaps = []
        tested = 0
        for k in idxs:
            i, j = candidates[k]
            mu_plus = prior.copy()
            mu_minus = prior.copy()
            # i->j 与 j->i 对称转移 eps
            mu_plus[i] -= eps; mu_plus[j] += eps
            mu_minus[j] -= eps; mu_minus[i] += eps
            # 简单检查
            if (mu_plus < -1e-12).any() or (mu_minus < -1e-12).any():
                continue
            # 精度修正
            mu_plus = np.maximum(mu_plus, 0.0); mu_plus /= mu_plus.sum()
            mu_minus = np.maximum(mu_minus, 0.0); mu_minus /= mu_minus.sum()

            v_plus, _ = self.welfare(mu_plus)
            v_minus, _ = self.welfare(mu_minus)
            gap = 0.5 * (v_plus + v_minus) - v0  # 凹函数应 ≤ 0
            if gap > 1e-10:
                viol_gaps.append(gap)
            tested += 1

        return dict(is_locally_concave=(len(viol_gaps) == 0),
                    n_pairs_tested=tested,
                    n_viol=len(viol_gaps),
                    max_gap=(max(viol_gaps) if viol_gaps else 0.0),
                    v0=v0, p0=p0)

    # ================== 解析判别所需的导数 & 核 K ==================
    def _phi_prime(self, theta: float, p: float) -> float:
        return -(theta - 1.0) * (1.0 - p) ** (theta - 2.0)

    def _vtheta_prime(self, theta: float, p: float) -> float:
        # 与证明一致：v_theta'(p)
        return (
            self.V * theta * (1.0 - p) ** (theta - 1.0)
            + self.R * (theta * (1.0 - p) ** (theta - 1.0)
                        - theta * (theta - 1.0) * p * (1.0 - p) ** (theta - 2.0))
            - self.c * theta
        )

    def _Bgp_at(self, mu: np.ndarray, p: float):
        theta = self.states.astype(float)
        v1 = np.array([self._vtheta_prime(t, p) for t in theta], dtype=float)
        phi1 = np.array([self._phi_prime(t, p) for t in theta], dtype=float)
        B = float(np.dot(mu, v1))
        Gp = float(np.dot(mu, phi1))  # 内点下 < 0
        return B, Gp, v1, phi1

    def kernel_K(self, mu: np.ndarray, theta_L: int, theta_H: int):
        """
        按解析判别：K = (B/Gp) * Δϕ' - 2 * Δv'。
        返回 K 及组成部件，供调试或导出。
        """
        v0, p = self.welfare(mu)               # 统一使用 p^*(mu)
        B, Gp, v1, phi1 = self._Bgp_at(mu, p)
        dphi1 = phi1[theta_H - 1] - phi1[theta_L - 1]
        dv1   = v1[theta_H - 1] - v1[theta_L - 1]
        K = (B / Gp) * dphi1 - 2.0 * dv1
        return dict(K=K, B=B, Gp=Gp, dphi1=dphi1, dv1=dv1, p=p, v0=v0)

    def scan_kernel(self, mu: np.ndarray, neighbours_only: bool = True) -> pd.DataFrame:
        """
        扫描核 K：默认仅扫描相邻对 (k, k+1)，更稳健；也可扫描所有成对。
        """
        pairs = []
        if neighbours_only:
            for t in range(1, self.n_max):
                pairs.append((t, t + 1))
        else:
            for i in range(1, self.n_max):
                for j in range(i + 1, self.n_max + 1):
                    pairs.append((i, j))
        rows = []
        for (L, H) in pairs:
            rows.append(dict(theta_L=L, theta_H=H, **self.kernel_K(mu, L, H)))
        df = pd.DataFrame(rows)
        df['K_positive'] = df['K'] > 0
        return df

    # ------------------ 二元信号：严格的 τ 与后验 ------------------
    def binary_split_result(self, prior: np.ndarray, theta_star: int):
        """
        低块：{1,...,θ*-1} ；高块：{θ*,...,n_max}
        稳健计算 tau_L, tau_H，并对 posterior 进行安全归一化。
        """
        prior = prior / prior.sum()
        idx = int(np.clip(theta_star, 1, self.n_max))

        # --- 稳健的块概率 ---
        tau_L_raw = float(prior[:idx - 1].sum()) if idx > 1 else 0.0
        tau_L = 0.0 if tau_L_raw < 1e-15 else tau_L_raw
        tau_H = 1.0 - tau_L
        if tau_H < 1e-15:    # 极端情形：右块几乎空
            tau_H = 0.0
            tau_L = 1.0

        # --- 构造后验 ---
        post_L = np.zeros_like(prior)
        post_H = np.zeros_like(prior)
        if tau_L > 0.0 and idx > 1:
            post_L[:idx - 1] = prior[:idx - 1] / tau_L
        if tau_H > 0.0:
            post_H[idx - 1:] = prior[idx - 1:] / tau_H

        # --- 安全归一化（防数值漂移）---
        if tau_L > 0.0:
            sL = post_L.sum()
            if not np.isfinite(sL) or sL <= 0:
                block = prior[:idx - 1]
                s = block.sum()
                post_L = np.zeros_like(prior)
                if s > 0:
                    post_L[:idx - 1] = block / s
            else:
                post_L /= sL

        if tau_H > 0.0:
            sH = post_H.sum()
            if not np.isfinite(sH) or sH <= 0:
                block = prior[idx - 1:]
                s = block.sum()
                post_H = np.zeros_like(prior)
                if s > 0:
                    post_H[idx - 1:] = block / s
            else:
                post_H /= sH

        # --- 计算福利 ---
        vL, pL = (np.nan, np.nan)
        if tau_L > 0.0:
            vL, pL = self.welfare(post_L)

        vH, pH = (np.nan, np.nan)
        if tau_H > 0.0:
            vH, pH = self.welfare(post_H)

        EV = (tau_L * (0.0 if np.isnan(vL) else vL) +
              tau_H * (0.0 if np.isnan(vH) else vH))

        return dict(
            threshold=idx,
            tau_L=float(tau_L), tau_H=float(tau_H),
            vL=float(vL) if not np.isnan(vL) else np.nan,
            vH=float(vH) if not np.isnan(vH) else np.nan,
            EV=float(EV),
            pL=float(pL) if not np.isnan(pL) else np.nan,
            pH=float(pH) if not np.isnan(pH) else np.nan
        )

    def best_binary_improvement(self, prior: np.ndarray):
        v0, p0 = self.welfare(prior)
        best = None
        per_threshold = []

        for th in range(1, self.n_max + 1):
            res = self.binary_split_result(prior, th)
            res['improvement'] = res['EV'] - v0
            res['improvement_pct'] = 100.0 * (res['improvement'] / abs(v0) if v0 != 0 else 0.0)
            per_threshold.append(res)
            if (best is None) or (res['improvement'] > best['improvement']):
                best = res

        best['v0'] = v0
        best['p0'] = p0
        return best, per_threshold

    # ------------------ 主入口：扫描 λ 并输出 ------------------
    def run(self, lam_values, outdir="results_kernel_binary_Kfinal", scan_all_pairs=False):
        os.makedirs(outdir, exist_ok=True)
        rows = []
        per_lambda_thr_tables = {}
        per_lambda_K_tables = {}

        print(f"参数：R={self.R:.1f}, c={self.c:.1f}, V={self.V:.1f}，c/R={self.c/self.R:.3f}")
        print("-" * 66)

        for lam in lam_values:
            prior = self.prior_poisson_trunc(lam)

            # 局部凹性（Jensen）
            loc = self.local_concavity_kernel(prior, eps=1e-3, n_pairs=80)

            # 解析核 K（相邻对/全对）
            ker = self.scan_kernel(prior, neighbours_only=not scan_all_pairs)
            any_K_pos = bool(ker['K_positive'].any())
            K_pos_pairs = int(ker['K_positive'].sum())

            # 二元信号
            best, thr_table = self.best_binary_improvement(prior)

            rows.append(dict(
                lam=lam,
                welfare_prior=loc['v0'],
                p_prior=loc['p0'],
                locally_concave=loc['is_locally_concave'],
                max_jensen_gap=loc['max_gap'],
                best_threshold=best['threshold'],
                tau_L=best['tau_L'], tau_H=best['tau_H'],
                vL=best['vL'], vH=best['vH'],
                EV=best['EV'],
                improvement=best['improvement'],
                improvement_pct=best['improvement_pct'],
                any_K_pos=any_K_pos,
                K_pos_pairs=K_pos_pairs
            ))
            per_lambda_thr_tables[lam] = pd.DataFrame(thr_table)
            per_lambda_K_tables[lam] = ker

            tag = "凹" if loc['is_locally_concave'] else "非凹"
            print(f"λ={lam:.1f}: {tag}，改进 {best['improvement_pct']:.2f}% ，θ*={best['threshold']}, "
                  f"τ_L={best['tau_L']:.3f}, τ_H={best['tau_H']:.3f} | 核>0 对数={K_pos_pairs}")

        df_main = pd.DataFrame(rows)

        # ====== 画图 ======
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))

        ax = axes[0, 0]
        colors = ['green' if b else 'red' for b in df_main['locally_concave']]
        ax.scatter(df_main['lam'], df_main['welfare_prior'], c=colors, s=50, alpha=0.8)
        ax.set_title('局部凹性（绿=凹，红=非凹）'); ax.set_xlabel('λ'); ax.set_ylabel('无信号福利 v(μ)'); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        corner_mask = (df_main['p_prior'] > 0.999)
        ax.scatter(df_main[~corner_mask]['lam'], df_main[~corner_mask]['p_prior'],
                   c='tab:blue', label='内点', s=50)
        ax.scatter(df_main[corner_mask]['lam'], df_main[corner_mask]['p_prior'],
                   c='tab:red', label='角点(p=1)', s=50)
        ax.set_title('均衡类型'); ax.set_xlabel('λ'); ax.set_ylabel('均衡响应率 p'); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 2]
        mu1 = [self.prior_poisson_trunc(l)[0] for l in df_main['lam']]
        ax.plot(df_main['lam'], mu1, label='μ(1)')
        ax.axhline(y=self.c/self.R, color='r', ls='--', label=f'c/R={self.c/self.R:.3f}')
        ax.set_title('μ(1) 与 c/R'); ax.set_xlabel('λ'); ax.set_ylabel('概率'); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(df_main['lam'], df_main['improvement_pct'], 'o-', color='tab:green')
        ax.axhline(0, color='k', ls='--', alpha=0.4)
        ax.set_title('最优二元改进（%）'); ax.set_xlabel('λ'); ax.set_ylabel('改进百分比'); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(df_main['lam'], df_main['max_jensen_gap'], 'o-', color='tab:orange')
        ax.set_title('Jensen 最大违背（非凹度量）'); ax.set_xlabel('λ'); ax.set_ylabel('max gap'); ax.grid(alpha=0.3)

        ax = axes[1, 2]
        ax.plot(df_main['lam'], df_main['best_threshold'], 'o-', color='tab:brown')
        ax.set_title('最优阈值 θ*'); ax.set_xlabel('λ'); ax.set_ylabel('θ*'); ax.grid(alpha=0.3)

        ax = axes[2, 0]
        counts = df_main['locally_concave'].value_counts()
        ax.bar(['凹', '非凹'], [counts.get(True, 0), counts.get(False, 0)],
               color=['tab:green', 'tab:red'])
        ax.set_title('局部凹性统计'); ax.grid(axis='y', alpha=0.3)

        ax = axes[2, 1]
        imp_mask = df_main['improvement'] > 1e-9
        ax.bar(['有改进', '无改进'], [imp_mask.sum(), (~imp_mask).sum()], color=['teal', 'gray'])
        ax.set_title('二元信号是否产生改进'); ax.grid(axis='y', alpha=0.3)

        ax = axes[2, 2]
        ax.plot(df_main['lam'], df_main['K_pos_pairs'], 'o-', color='tab:purple')
        ax.set_title('解析核 K>0 的对数'); ax.set_xlabel('λ'); ax.set_ylabel('#(K>0)'); ax.grid(alpha=0.3)

        plt.suptitle('VOD：显式核 𝒦 的凹性与信息价值（最终版）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        png_path = os.path.join(outdir, 'summary_kernel_binary_Kfinal.png')
        plt.savefig(png_path, dpi=220, bbox_inches='tight')
        plt.close()

        # ====== 导出 Excel ======
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        xlsx_path = os.path.join(outdir, f'results_kernel_binary_Kfinal_{ts}.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            # 主结果
            df_main.to_excel(writer, sheet_name='Main', index=False)
            # 每个 λ 的阈值细表
            for lam in lam_values:
                thr_df = per_lambda_thr_tables[lam]
                sheet = f"thr_lambda_{lam}".replace('.', '_')
                sheet = sheet[:31]
                thr_df.to_excel(writer, sheet_name=sheet, index=False)
            # 每个 λ 的核表
            for lam in lam_values:
                ker_df = per_lambda_K_tables[lam]
                sheet = f"kernel_lambda_{lam}".replace('.', '_')
                sheet = sheet[:31]
                ker_df.to_excel(writer, sheet_name=sheet, index=False)

        print(f"\n✅ 图已保存：{png_path}")
        print(f"✅ Excel 已保存：{xlsx_path}")
        return df_main, per_lambda_thr_tables, per_lambda_K_tables, png_path, xlsx_path


def main():
    analyzer = VODKernelBinaryAnalyzer(n_max=30, R=10.0, c=1.0, V=50.0)
    lam_values = np.arange(1, 21, 1.0)  # λ=1..20
    df_main, thr_tabs, K_tabs, png, xlsx = analyzer.run(lam_values, scan_all_pairs=False)
    # 控制台小结
    any_tau_issue = ((df_main['tau_L'] < -1e-12) | (df_main['tau_L'] > 1+1e-12) |
                     (df_main['tau_H'] < -1e-12) | (df_main['tau_H'] > 1+1e-12)).any()
    print("\nτ 检查：", "通过（均在 [0,1] 且和为 1）" if not any_tau_issue else "发现异常！")
    print("核 K>0 是否与“有改进”一致？",
          "是" if (df_main['any_K_pos'] == (df_main['improvement'] > 1e-9)).all() else "存在差异（请核查阈值/数值容差）")
    return df_main


if __name__ == "__main__":
    _ = main()