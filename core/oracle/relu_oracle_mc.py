# precision_estimation/core/oracle/relu_oracle_mc.py
"""
DataAwareMCReLUOracle (tight, optimized, GPU-safe RNG)

- 预计算并复用 y_ref/x_q/x_c/y_c
- zero-/exact-aware 输出噪声，且在 y_c 上注噪
- ReLU 默认关闭输入计算噪声
- ✅ 修正：所有 torch.Generator 都跟随相关张量的 device 创建
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from core.config.precision_strategy import (
    PrecisionStrategy,
    ulp_like,
    apply_input_quant,
    apply_output_quant,
    promote_exact,
)

@dataclass
class OracleResult:
    predicted_bound: float
    quantile: float
    safety_factor: float
    sample_size: int
    max_errors: List[float]
    component_estimates: Dict[str, float]

class DataAwareMCReLUOracle:
    def __init__(
        self,
        strategy: PrecisionStrategy,
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        enable_noise_input: bool = False,
        enable_noise_output: bool = True,
        component_samples: int = 32,
    ):
        self.strategy = strategy
        self.num_mc_samples = int(num_mc_samples)
        self.quantile = float(quantile)
        self.safety_factor = float(safety_factor)
        self.seeded = bool(seeded)
        if devices is None:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.devices = devices
        self.enable_noise_input = bool(enable_noise_input)
        self.enable_noise_output = bool(enable_noise_output)
        self.component_samples = int(component_samples)

    # ---- utils ----
    def _make_gen(self, device: torch.device, seed: Optional[int] = None) -> torch.Generator:
        """Create RNG on the same device as tensors."""
        dev = device if device.type == "cuda" else "cpu"
        g = torch.Generator(device=dev)
        if self.seeded and seed is not None:
            g.manual_seed(seed)
        return g

    # ---------- 预计算固定部分 ----------
    @torch.inference_mode()
    def _prepare_base(self, x: torch.Tensor):
        s = self.strategy
        y_ref = F.relu(x.double()).float()
        x_q = apply_input_quant(x, s)
        x_c = promote_exact(x_q, s.compute_dtype)

        if self.enable_noise_input:
            ulp_in = ulp_like(x_c, s.compute_dtype).to(device=x.device)
            g = self._make_gen(x.device, seed=202431)
            r = torch.rand(x_c.shape, device=x.device, dtype=x_c.dtype, generator=g)
            x_c = x_c + (r - 0.5) * ulp_in

        y_c = F.relu(x_c)
        return y_ref, x_q, x_c, y_c

    # ---------- 在 y_c 上做一次随机注噪 ----------
    @torch.inference_mode()
    def _sample_once_on_yc(self, y_c: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        s = self.strategy
        if not self.enable_noise_output:
            return apply_output_quant(y_c, s)

        ulp_out = ulp_like(y_c, s.output_dtype).to(device=y_c.device)
        y_round = y_c.to(dtype=s.output_dtype)
        y_back  = y_round.to(dtype=y_c.dtype)
        exact   = (y_back == y_c)
        is_zero = (y_round == 0)
        need_noise = (~exact) & (~is_zero)

        if need_noise.any():
            idx = need_noise.view(-1).nonzero(as_tuple=False).view(-1)
            ulp_flat = ulp_out.view(-1)
            y_flat   = y_c.view(-1)
            r = torch.rand(idx.numel(), device=y_c.device, dtype=y_c.dtype, generator=g)
            noise_flat = torch.zeros_like(y_flat)
            noise_flat[idx] = (r - 0.5) * ulp_flat[idx]
            y_c = (y_flat + noise_flat).view_as(y_c)

        y_out = apply_output_quant(y_c, s)
        return y_out

    # ---------- Monte-Carlo ----------
    @torch.inference_mode()
    def _mc_errors(self, y_c: torch.Tensor, y_ref: torch.Tensor) -> List[float]:
        g = self._make_gen(y_c.device, seed=1234)
        errs: List[float] = []
        for _ in range(self.num_mc_samples):
            y_out = self._sample_once_on_yc(y_c, g)
            err = (y_out - y_ref).abs().max().item()
            errs.append(float(err))
        return errs

    # ---------- 组件中位数 ----------
    @torch.inference_mode()
    def _component_median(self, x: torch.Tensor, which: str) -> float:
        s = self.strategy
        y_ref, x_q, x_c, y_c = self._prepare_base(x)

        g = self._make_gen(x.device, seed=4321)
        errs: List[float] = []
        n = min(self.component_samples, max(1, self.num_mc_samples // 4))

        if which == 'input':
            ulp_in = ulp_like(x_c, s.compute_dtype).to(device=x.device)
            for _ in range(n):
                r = torch.rand(x_c.shape, device=x.device, dtype=x_c.dtype, generator=g)
                x_c_noisy = x_c + (r - 0.5) * ulp_in
                y_c_noisy = F.relu(x_c_noisy)
                y_out = apply_output_quant(y_c_noisy, s)
                errs.append(float((y_out - y_ref).abs().max().item()))
        else:  # 'output'
            for _ in range(n):
                y_out = self._sample_once_on_yc(y_c, g)
                errs.append(float((y_out - y_ref).abs().max().item()))

        if not errs:
            return 0.0
        return float(torch.median(torch.tensor(errs, dtype=torch.float32)).item())

    # ---------- Public API ----------
    @torch.inference_mode()
    def predict_error_bound(self, x: torch.Tensor) -> OracleResult:
        y_ref, x_q, x_c, y_c = self._prepare_base(x)
        max_errors = self._mc_errors(y_c, y_ref)
        qv = torch.quantile(torch.tensor(max_errors), q=self.quantile).item()
        bound = float(qv * self.safety_factor)
        comp = {
            "input_storage_error": self._component_median(x, 'input'),
            "output_storage_error": self._component_median(x, 'output'),
        }
        return OracleResult(
            predicted_bound=bound,
            quantile=self.quantile,
            safety_factor=self.safety_factor,
            sample_size=len(max_errors),
            max_errors=max_errors,
            component_estimates=comp,
        )
