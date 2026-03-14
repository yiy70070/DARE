# precision_estimation/core/analyzer/gemm_error_analyzer.py
"""
GEMMErrorAnalyzer (aligned with new GEMM oracle)
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
)

class GEMMErrorAnalyzer:
    def __init__(self, accum_k_cap: int = 64):
        self.accum_k_cap = int(accum_k_cap)

    def _effective_accum_dtype(self, strategy: PrecisionStrategy) -> torch.dtype:
        if strategy.compute_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return strategy.compute_dtype

    def _accum_sigma(self, strategy: PrecisionStrategy, accum_dtype: torch.dtype) -> float:
        return 0.08 if (accum_dtype == torch.float32 and strategy.compute_dtype in (torch.float16, torch.bfloat16)) else 0.5

    def analyze(
        self,
        oracle_result: Any,
        x: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        strategy: Optional[PrecisionStrategy] = None,
        gemm_params: Optional[Dict[str, bool]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        comp_est = oracle_result.component_estimates if hasattr(oracle_result, "component_estimates") else {}
        comp_est = {k: float(v) for k, v in comp_est.items()} if comp_est else {}
        total_pred = sum(comp_est.values()) if comp_est else None
        comp_ratios = {}
        primary = None
        if comp_est and total_pred and total_pred > 0:
            for k, v in comp_est.items():
                comp_ratios[k] = v / total_pred
            primary = max(comp_ratios.items(), key=lambda kv: kv[1])[0]
        elif comp_est:
            comp_ratios = {k: 0.0 for k in comp_est}
            primary = max(comp_est.items(), key=lambda kv: kv[1])[0] if comp_est else None

        report: Dict[str, Any] = {
            "component_estimates": comp_est,
            "component_ratios": comp_ratios,
            "primary_source": primary,
            "top_elements": [],
            "suggestion": "",
        }

        if x is None or w is None or strategy is None:
            report["suggestion"] = self._suggest_from_primary(primary)
            return report

        x = x.detach().cpu().to(dtype=torch.float32)
        w = w.detach().cpu().to(dtype=torch.float32)

        if gemm_params is None:
            gemm_params = {"transpose_a": False, "transpose_b": False}

        with torch.no_grad():
            # reference
            x_ref = x.double(); w_ref = w.double()
            if gemm_params.get("transpose_a", False): x_ref = x_ref.T
            if gemm_params.get("transpose_b", False): w_ref = w_ref.T
            y_ref = torch.mm(x_ref, w_ref).to(dtype=torch.float32)

            # mixed
            x_q = apply_input_quant(x, strategy)
            w_q = apply_weight_quant(w, strategy)
            x_c = x_q.to(dtype=strategy.compute_dtype)
            w_c = w_q.to(dtype=strategy.compute_dtype)
            if gemm_params.get("transpose_a", False): x_c = x_c.T
            if gemm_params.get("transpose_b", False): w_c = w_c.T
            y_c = torch.mm(x_c, w_c)
            y_mixed = apply_output_quant(y_c, strategy)

            per_elem_err = (y_ref - y_mixed).abs()
            report["max_element_error"] = float(per_elem_err.max().item())

            M, N = y_ref.shape
            K = x_c.shape[1]
            report["matrix_dims"] = {"M": M, "N": N, "K": K}

            flat = per_elem_err.view(-1)
            topk = min(top_k, flat.numel())
            if topk > 0:
                topk_vals, topk_idx = torch.topk(flat, topk)
                coords = [(idx // N, idx % N) for idx in topk_idx.tolist()]

                accum_dtype = self._effective_accum_dtype(strategy)
                sigma = self._accum_sigma(strategy, accum_dtype)
                k_eff = min(int(K), self.accum_k_cap)

                for (i, j) in coords:
                    elem_error = float(per_elem_err[i, j].item())
                    y_ref_val = float(y_ref[i, j].item())
                    y_mixed_val = float(y_mixed[i, j].item())

                    x_row = x[i, :] if not gemm_params.get("transpose_a", False) else x[:, i]
                    w_col = w[:, j] if not gemm_params.get("transpose_b", False) else w[j, :]

                    x_row_q = x_q[i, :] if not gemm_params.get("transpose_a", False) else x_q[:, i]
                    w_col_q = w_q[:, j] if not gemm_params.get("transpose_b", False) else w_q[j, :]

                    x_storage_err = (x_row - x_row_q).abs().sum().item()
                    w_storage_err = (w_col - w_col_q).abs().sum().item()

                    y_c_val = float(y_c[i, j].item())
                    ulp_est = ulp_like(torch.tensor([y_c_val], dtype=torch.float32), accum_dtype).item()
                    accum_err_est = math.sqrt(k_eff) * sigma * ulp_est
                    demote_err = abs(y_c_val - y_mixed_val)

                    products = x_row_q.to(dtype=strategy.compute_dtype) * w_col_q.to(dtype=strategy.compute_dtype)
                    products_f32 = products.to(dtype=torch.float32)

                    elem_record = {
                        "coord": (i, j),
                        "element_error": elem_error,
                        "y_ref": y_ref_val,
                        "y_mixed": y_mixed_val,
                        "x_storage_error": x_storage_err,
                        "w_storage_error": w_storage_err,
                        "accumulation_error_estimate": accum_err_est,
                        "demote_error": demote_err,
                        "accumulation_dim": K,
                        "condition_indicator": self._estimate_condition_number(x_row_q, w_col_q),
                        "product_stats": {
                            "min": float(products_f32.min().item()),
                            "max": float(products_f32.max().item()),
                            "mean": float(products_f32.mean().item()),
                            "std": float(products_f32.std().item()),
                        }
                    }
                    report["top_elements"].append(elem_record)

            if report["top_elements"]:
                total_x = sum(e["x_storage_error"] for e in report["top_elements"])
                total_w = sum(e["w_storage_error"] for e in report["top_elements"])
                total_a = sum(e["accumulation_error_estimate"] for e in report["top_elements"])
                total_d = sum(e["demote_error"] for e in report["top_elements"])
                s = total_x + total_w + total_a + total_d + 1e-20
                agg = {"x_storage": total_x/s, "w_storage": total_w/s, "accumulation": total_a/s, "demote": total_d/s}
                report["aggregated_ratios"] = agg
                report["primary_source_refined"] = max(agg.items(), key=lambda kv: kv[1])[0]

                avg_cond = sum(e["condition_indicator"] for e in report["top_elements"]) / len(report["top_elements"])
                report["numerical_stability"] = {
                    "average_condition_indicator": avg_cond,
                    "accumulation_dimension": K,
                    "stability_assessment": self._assess_stability(K, avg_cond)
                }
                report["suggestion"] = self._suggest_from_gemm_analysis(report["primary_source_refined"], K, avg_cond)
            else:
                report["suggestion"] = self._suggest_from_primary(primary)

        return report

    def _estimate_condition_number(self, x_vec: torch.Tensor, w_vec: torch.Tensor) -> float:
        try:
            x_abs = x_vec.abs(); w_abs = w_vec.abs()
            x_max, x_min = x_abs.max().item(), x_abs.min().item() + 1e-12
            w_max, w_min = w_abs.max().item(), w_abs.min().item() + 1e-12
            return float(math.log10(max(x_max/x_min, w_max/w_min)))
        except:
            return 0.0

    def _assess_stability(self, K: int, avg_condition: float) -> str:
        if K > 2048 and avg_condition > 4: return "poor"
        elif K > 1024 and avg_condition > 3: return "moderate"
        elif avg_condition > 5: return "poor"
        else: return "good"

    def _suggest_from_primary(self, primary: Optional[str]) -> str:
        if primary is None:
            return "Unable to determine primary error source; please provide more runtime context for detailed diagnosis."
        if "demote" in primary or "output" in primary:
            return "Primary error source is output demotion. Suggestions: delay precision reduction or increase output precision; consider block-wise computation to maintain high-precision accumulation in large matrix multiplications."
        if "accum" in primary:
            return "Primary error source is accumulation rounding. Suggestions: use Kahan summation or higher precision accumulation; consider matrix blocking strategy to reduce single accumulation length; check accumulation dimension size."
        if "input" in primary or "x_storage" in primary:
            return "Primary error source is input storage precision. Suggestions: increase input storage precision; perform input normalization or weight scaling to improve numerical range."
        if "weight" in primary or "w_storage" in primary:
            return "Primary error source is weight storage precision. Suggestions: maintain weights in high precision (FP32); use more precise weight quantization strategies; check if weight distribution is uniform."
        return "Analysis result is unclear; please check input/weight distribution or increase MC sampling; GEMM is sensitive to accumulation dimension and condition number."

    def _suggest_from_gemm_analysis(self, primary: str, K: int, avg_condition: float) -> str:
        base = self._suggest_from_primary(primary)
        if K > 2048: base += f" Accumulation dimension K={K} is large; consider tiling to reduce single accumulation length or use higher precision accumulator."
        if avg_condition > 4: base += f" Numerical stability issue (indicator={avg_condition:.2f}). Consider normalization or check for ill-conditioned matrices."
        if "accumulation" in primary and K > 1024: base += " For large K: consider FP32 accumulation with low-precision inputs."
        return base
