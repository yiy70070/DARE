# precision_estimation/core/detector/matmul_error_detector.py
from typing import Tuple
import torch

from core.config.precision_strategy import (
    PrecisionStrategy,
    apply_input_quant,
    apply_weight_quant,
    apply_output_quant,
    promote_exact,
)
from core.oracle.matmul_oracle_mc import DataAwareMCMatmulOracle, OracleResult


class MatmulErrorDetector:


    def __init__(self, strategy: PrecisionStrategy, oracle: DataAwareMCMatmulOracle):
        self.strategy = strategy
        self.oracle = oracle

    @torch.inference_mode()
    def _actual_run(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        s = self.strategy
        A_q = apply_input_quant(A, s)
        B_q = apply_weight_quant(B, s)
        A_c = promote_exact(A_q, s.compute_dtype)
        B_c = promote_exact(B_q, s.compute_dtype)
        Y_c = torch.matmul(A_c, B_c)
        Y_out = apply_output_quant(Y_c, s)
        return Y_out

    @torch.inference_mode()
    def detect(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[bool, float, float, OracleResult]:
        oracle_result: OracleResult = self.oracle.predict_error_bound(A, B)
        bound = oracle_result.predicted_bound

        Y_ref = torch.matmul(A.double(), B.double()).float()
        Y_act = self._actual_run(A, B)
        err = (Y_act - Y_ref).abs().max().item()
        exceeded = bool(err > bound)
        return exceeded, err, bound, oracle_result
