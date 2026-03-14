# precision_estimation/core/analyzer/matmul_error_analyzer.py
"""
MatmulErrorAnalyzer

Interpretable analysis of mixed-precision errors for A @ B (structure consistent with Conv2D version):
- Returns component_estimates / ratios / primary_source
- If A, B, strategy are provided, performs critical path decomposition for top-k output elements (i,j):
  ref[i,j] = sum_k A[i,k]*B[k,j]
  Decomposes error into:
    - A storage rounding (input storage)
    - B storage rounding (weight storage)
    - Accumulation rounding (accumulation)
    - Output demotion (demote) —— distributed by contribution ratio
"""
from typing import Any, Dict, Optional, Tuple, List
import math
import torch

from core.config.precision_strategy import (
    ulp_like,
    apply_input_quant,
    apply_weight_quant,
    apply_output_quant,
    PrecisionStrategy,
    demote_with_round,
    quantize_to_dtype,
)


class MatmulErrorAnalyzer:
    def __init__(self):
        pass

    def analyze(
        self,
        oracle_result: Any,
        A: Optional[torch.Tensor] = None,
        B: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        comp_est = getattr(oracle_result, "component_estimates", {}) or {}
        comp_est = {k: float(v) for k, v in comp_est.items()}
        total_pred = sum(comp_est.values()) if comp_est else None
        comp_ratios = {}
        primary = None
        if comp_est and total_pred and total_pred > 0:
            comp_ratios = {k: v / total_pred for k, v in comp_est.items()}
            primary = max(comp_ratios.items(), key=lambda kv: kv[1])[0]
        elif comp_est:
            comp_ratios = {k: 0.0 for k in comp_est}
            primary = max(comp_est.items(), key=lambda kv: kv[1])[0] if comp_est else None

        report: Dict[str, Any] = {
            "component_estimates": comp_est,
            "component_ratios": comp_ratios,
            "primary_source": primary,
            "top_paths": [],
            "suggestion": "",
        }

        if A is None or B is None or strategy is None:
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        A = A.detach().cpu().to(dtype=torch.float32)
        B = B.detach().cpu().to(dtype=torch.float32)

        with torch.no_grad():
            Y_ref = torch.matmul(A.double(), B.double()).to(dtype=torch.float32)

            # Mixed precision path
            A_q = apply_input_quant(A, strategy)
            B_q = apply_weight_quant(B, strategy)
            A_c = A_q.to(dtype=strategy.compute_dtype)
            B_c = B_q.to(dtype=strategy.compute_dtype)
            Y_c = torch.matmul(A_c, B_c)
            Y_mixed = apply_output_quant(Y_c, strategy)

            per_elem_err = (Y_ref - Y_mixed).abs()
            flat = per_elem_err.view(-1)
            num_out = flat.numel()
            if num_out == 0:
                report["suggestion"] = "Empty output."
                return report

            topk = min(int(top_k), num_out)
            topk_vals, topk_idx = torch.topk(flat, topk)
            M, N = Y_ref.shape

            coords: List[Tuple[int, int]] = []
            for idx in topk_idx.tolist():
                i = idx // N
                j = idx % N
                coords.append((i, j))

            A_q_full = A_q
            B_q_full = B_q
            for i, j in coords:
                # ref[i,j] = sum_k A[i,k] * B[k,j]
                A_row = A[i, :]                   # [K]
                B_col = B[:, j]                   # [K]
                K = A_row.numel()

                # Storage rounding errors
                A_q_row = A_q_full[i, :]
                B_q_col = B_q_full[:, j]
                eps_A = (A_q_row - A_row).abs()
                eps_B = (B_q_col - B_col).abs()

                # Element-wise products
                prod_q = (A_q_row.to(dtype=strategy.compute_dtype) * B_q_col.to(dtype=strategy.compute_dtype)).to(dtype=torch.float32)
                sum_abs_prod = prod_q.abs().sum().clamp(min=1e-12)

                # Accumulation and demotion
                y_c = prod_q.sum().to(dtype=strategy.compute_dtype).to(dtype=torch.float32)
                y_demoted = apply_output_quant(y_c.unsqueeze(0), strategy).squeeze(0)
                demote_err_pixel = (y_demoted - y_c).abs().item()

                ulp_y = ulp_like(y_c.unsqueeze(0), strategy.compute_dtype).squeeze(0).item()
                accum_total = 0.5 * ulp_y * math.sqrt(max(1, K))

                # Per-element contributions
                abs_B = B_col.abs()
                abs_A = A_row.abs()
                linear_term = (abs_B * eps_A) + (abs_A * eps_B)     # [K]
                abs_prod = prod_q.abs()
                accum_share = (abs_prod / sum_abs_prod) * accum_total  # [K]

                contrib_vec = linear_term + accum_share
                sum_contrib = contrib_vec.sum().clamp(min=1e-12)
                demote_share_vec = (contrib_vec / sum_contrib) * demote_err_pixel
                total_elem_contrib = contrib_vec + demote_share_vec

                pixel_error = float(per_elem_err[i, j].item())
                yref_val = float(Y_ref[i, j].item())
                ymix_val = float(Y_mixed[i, j].item())

                contrib_list = []
                for k in range(K):
                    c_val = float(total_elem_contrib[k].item())
                    percent = c_val / (pixel_error + 1e-20)
                    contrib_list.append({
                        "a_coord": (i, k),
                        "b_coord": (k, j),
                        "product": float(prod_q[k].item()),
                        "linear_term": float(linear_term[k].item()),
                        "accum_share": float(accum_share[k].item()),
                        "demote_share": float(demote_share_vec[k].item()),
                        "contribution": c_val,
                        "contribution_ratio_of_pixel": percent,
                    })
                contrib_list.sort(key=lambda x: x["contribution"], reverse=True)

                report["top_paths"].append({
                    "out_coord": (i, j),
                    "pixel_error": pixel_error,
                    "y_ref": yref_val,
                    "y_mixed": ymix_val,
                    "demote_error_pixel": demote_err_pixel,
                    "accum_estimate": accum_total,
                    "num_contributing_elements": K,
                    "top_contributors": contrib_list[: min(10, len(contrib_list))],
                })

            # Aggregate ratios, refine primary source
            agg = {"linear": 0.0, "accum": 0.0, "demote": 0.0}
            for p in report["top_paths"]:
                for c in p["top_contributors"]:
                    agg["linear"] += c["linear_term"]
                    agg["accum"] += c["accum_share"]
                    agg["demote"] += c["demote_share"]
            tot = sum(abs(v) for v in agg.values()) + 1e-20
            agg_ratio = {k: float(abs(v) / tot) for k, v in agg.items()}
            report["aggregated_contributions"] = agg
            report["aggregated_ratios"] = agg_ratio
            primary_refined = max(agg_ratio.items(), key=lambda kv: kv[1])[0] if agg_ratio else primary
            report["primary_source_refined"] = primary_refined
            report["suggestion"] = self._suggest_from_aggregated(primary_refined, comp_ratios)

        return report

    # ---------- helpers ----------
    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        """Generate suggestions based on primary error source"""
        if primary is None:
            return "Unable to determine primary error source; please provide A/B/strategy for fine-grained diagnosis."
        if "demote" in primary or "output" in primary:
            return "Primary error source is output demotion. Suggestions: delay precision reduction or increase output precision; use FP32 storage for critical path locations."
        if "accum" in primary:
            return "Primary error source is accumulation rounding. Suggestions: increase accumulation precision, use Kahan/block-wise accumulation algorithms."
        if "input" in primary or "storage" in primary:
            return "Primary error source is input storage (A). Suggestions: increase A's storage precision or normalize to compress dynamic range."
        if "weight" in primary:
            return "Primary error source is weight storage (B). Suggestions: maintain B in FP32 or optimize quantization strategy."
        return "Analysis result is unclear; please increase MC sampling or check data distribution."

    def _suggest_from_aggregated(self, primary: Optional[str], comp_ratios: Dict[str, float]) -> str:
        """Generate suggestions based on aggregated analysis"""
        base = self._suggest_from_primary(primary)
        if comp_ratios:
            sorted_comp = sorted(comp_ratios.items(), key=lambda kv: kv[1], reverse=True)
            top_name, top_ratio = sorted_comp[0]
            if top_ratio > 0.6 and (primary is None or top_name not in primary):
                base += f" Note: Global estimate shows `{top_name}` also has high proportion ({top_ratio:.2%})."
        return base
