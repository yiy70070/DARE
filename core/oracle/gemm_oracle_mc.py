# precision_estimation/core/oracle/gemm_oracle_mc.py
"""
DataAwareMCGEMMOracle (tight but realistic)

Key changes:
- Output noise is zero-/exact-aware and injected w.r.t. pre-quantized y_c, then quantized.
- Accumulation noise models FP32-accumulation by default when compute is FP16/BF16:
    accum_dtype = 'auto' -> torch.float32 if compute in {fp16, bf16} else compute
    k_eff = min(K, accum_k_cap)   # default 64 (tile-like)
    noise ~ (r-0.5) * ULP(accum_dtype) * sqrt(k_eff) * sigma
      where sigma = accum_noise_scale_fp32 (default 0.08) for FP32-accum
                    accum_noise_scale_same (default 0.5)  when accum==compute
- Input/weight extra storage noise default OFF (to avoid double counting).
- Multi-GPU worker design retained; RNGs created on the same device as tensors.
"""

import os
import math
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from core.config.precision_strategy import (
    PrecisionStrategy,
    ulp_like,
    apply_input_quant,
    apply_weight_quant,
    apply_output_quant,
)

@dataclass
class OracleResult:
    predicted_bound: float
    quantile: float
    safety_factor: float
    sample_errors: List[float]
    component_estimates: Dict[str, float]
    meta: Dict[str, Any]


class DataAwareMCGEMMOracle:
    def __init__(
        self,
        strategy: PrecisionStrategy,
        gemm_params: Dict[str, Any],
        num_mc_samples: int = 512,
        quantile: float = 0.999,
        safety_factor: float = 1.10,
        seeded: bool = True,
        devices: Optional[List[int]] = None,
        # noise toggles (default: avoid double counting on storage sides)
        enable_noise_input: bool = False,
        enable_noise_weight: bool = False,
        enable_noise_accum: bool = True,
        enable_noise_output: bool = True,
        # accumulation modeling
        accum_dtype: Union[str, torch.dtype] = "auto",
        accum_k_cap: int = 64,
        accum_noise_scale_same: float = 0.5,   # when accum dtype == compute dtype
        accum_noise_scale_fp32: float = 0.08,  # when FP32-accum with fp16/bf16 compute
    ):
        self.strategy = strategy
        self.params = {
            "transpose_a": gemm_params.get("transpose_a", False),
            "transpose_b": gemm_params.get("transpose_b", False),
        }
        self.num_mc_samples = int(num_mc_samples)
        self.quantile = float(quantile)
        self.safety_factor = float(safety_factor)
        self.seeded = bool(seeded)

        if devices is None:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.devices = devices

        self.enable_noise_input = bool(enable_noise_input)
        self.enable_noise_weight = bool(enable_noise_weight)
        self.enable_noise_accum = bool(enable_noise_accum)
        self.enable_noise_output = bool(enable_noise_output)

        self.accum_dtype_cfg = accum_dtype
        self.accum_k_cap = int(accum_k_cap)
        self.accum_noise_scale_same = float(accum_noise_scale_same)
        self.accum_noise_scale_fp32 = float(accum_noise_scale_fp32)

    # ---------- helpers ----------
    @staticmethod
    @torch.inference_mode()
    def _compute_reference_on_device(x_cpu: torch.Tensor, w_cpu: torch.Tensor, device: torch.device, params: Dict[str, Any]) -> torch.Tensor:
        x64 = x_cpu.to(device=device, dtype=torch.float64)
        w64 = w_cpu.to(device=device, dtype=torch.float64)
        if params.get("transpose_a", False):
            x64 = x64.T
        if params.get("transpose_b", False):
            w64 = w64.T
        return torch.mm(x64, w64).to(dtype=torch.float32)

    def _effective_accum_dtype(self) -> torch.dtype:
        if isinstance(self.accum_dtype_cfg, torch.dtype):
            return self.accum_dtype_cfg
        # 'auto'
        if self.strategy.compute_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return self.strategy.compute_dtype

    def _accum_sigma(self, eff_dtype: torch.dtype) -> float:
        if eff_dtype == torch.float32 and self.strategy.compute_dtype in (torch.float16, torch.bfloat16):
            return self.accum_noise_scale_fp32
        return self.accum_noise_scale_same

    # ---------- worker ----------
    @staticmethod
    def _worker_run(
        rank: int,
        device_id: Optional[int],
        x_cpu: torch.Tensor,
        w_cpu: torch.Tensor,
        params: Dict[str, Any],
        strategy: PrecisionStrategy,
        num_local: int,
        noise_mask: Tuple[bool, bool, bool, bool],
        seed_base: int,
        return_queue: mp.Queue,
        accum_dtype_eff: torch.dtype,
        accum_k_cap: int,
        accum_sigma_same: float,
        accum_sigma_fp32: float,
    ):
        try:
            torch.set_num_threads(max(1, os.cpu_count() // 8))
            use_cuda = (device_id is not None) and torch.cuda.is_available()
            device = torch.device(f"cuda:{device_id}") if use_cuda else torch.device("cpu")
            if use_cuda:
                try:
                    torch.cuda.set_device(device)
                except Exception:
                    pass

            # reference on worker device
            y_ref = DataAwareMCGEMMOracle._compute_reference_on_device(x_cpu, w_cpu, device, params)

            base_seed = int(seed_base) + 1337 * (rank + 1)
            errors: List[float] = []

            # choose sigma for accumulation
            if accum_dtype_eff == torch.float32 and strategy.compute_dtype in (torch.float16, torch.bfloat16):
                accum_sigma = accum_sigma_fp32
            else:
                accum_sigma = accum_sigma_same

            for i in range(num_local):
                g = torch.Generator(device=device if device.type == 'cuda' else 'cpu')
                g.manual_seed(base_seed + i)

                # move inputs
                x = x_cpu.to(device=device, dtype=torch.float32)
                w = w_cpu.to(device=device, dtype=torch.float32)

                # storage quantization (deterministic)
                x_q = apply_input_quant(x, strategy)
                w_q = apply_weight_quant(w, strategy)

                # promote to compute
                x_c = x_q.to(dtype=strategy.compute_dtype)
                w_c = w_q.to(dtype=strategy.compute_dtype)

                if params.get("transpose_a", False):
                    x_c = x_c.T
                if params.get("transpose_b", False):
                    w_c = w_c.T

                # (optional) extra storage noise -- generally OFF to avoid double counting
                if noise_mask[0]:
                    ulp_x = ulp_like(x_c, strategy.input_dtype).to(device=device)
                    r = torch.rand(x_c.shape, device=device, dtype=x_c.dtype, generator=g)
                    x_c = x_c + (r - 0.5) * ulp_x
                if noise_mask[1]:
                    ulp_w = ulp_like(w_c, strategy.weight_dtype).to(device=device)
                    r = torch.rand(w_c.shape, device=device, dtype=w_c.dtype, generator=g)
                    w_c = w_c + (r - 0.5) * ulp_w

                # GEMM
                y_c = torch.mm(x_c, w_c)

                # accumulation noise (tile-aware, dtype-aware)
                if noise_mask[2]:
                    K = x_c.shape[1]
                    k_eff = min(int(K), int(accum_k_cap))
                    ulp_acc = ulp_like(y_c, accum_dtype_eff).to(device=device)
                    r = torch.rand(y_c.shape, device=device, dtype=y_c.dtype, generator=g)
                    y_c = y_c + (r - 0.5) * ulp_acc * (math.sqrt(k_eff) * float(accum_sigma))

                # output noise (zero/exact-aware, inject w.r.t. y_c, then quantize)
                if noise_mask[3]:
                    ulp_out = ulp_like(y_c, strategy.output_dtype).to(device=device)
                    y_round = y_c.to(dtype=strategy.output_dtype)
                    y_back  = y_round.to(dtype=y_c.dtype)
                    exact   = (y_back == y_c)
                    r = torch.rand(y_c.shape, device=device, dtype=y_c.dtype, generator=g)
                    noise = (r - 0.5) * ulp_out
                    noise = torch.where(exact, torch.zeros_like(noise), noise)
                    y_out = apply_output_quant(y_c + noise, strategy)
                else:
                    y_out = apply_output_quant(y_c, strategy)

                err = (y_out - y_ref).abs().max().item()
                errors.append(err)

            return_queue.put((rank, errors, None))

        except Exception as e:
            tb = traceback.format_exc()
            return_queue.put((rank, [], f"{repr(e)}\n{tb}"))

    # ---------- driver ----------
    def predict_error_bound(self, x: torch.Tensor, w: torch.Tensor) -> OracleResult:
        x_cpu = x.detach().contiguous().cpu()
        w_cpu = w.detach().contiguous().cpu()

        noise_mask = (
            self.enable_noise_input,
            self.enable_noise_weight,
            self.enable_noise_accum,
            self.enable_noise_output,
        )

        accum_dtype_eff = self._effective_accum_dtype()
        q = mp.Queue()
        all_errors: List[float] = []

        if len(self.devices) == 0:
            self._worker_run(
                rank=0, device_id=None, x_cpu=x_cpu, w_cpu=w_cpu, params=self.params, strategy=self.strategy,
                num_local=self.num_mc_samples, noise_mask=noise_mask, seed_base=1234 if self.seeded else int(time.time()),
                return_queue=q, accum_dtype_eff=accum_dtype_eff, accum_k_cap=self.accum_k_cap,
                accum_sigma_same=self.accum_noise_scale_same, accum_sigma_fp32=self.accum_noise_scale_fp32,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Worker error: {err_msg}")
            all_errors = errors
        else:
            per = math.ceil(self.num_mc_samples / max(1, len(self.devices)))
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            procs = []
            for rank, dev in enumerate(self.devices):
                p = ctx.Process(
                    target=self._worker_run,
                    args=(
                        rank, dev, x_cpu, w_cpu, self.params, self.strategy, per, noise_mask,
                        1234 if self.seeded else int(time.time()), q,
                        accum_dtype_eff, self.accum_k_cap, self.accum_noise_scale_same, self.accum_noise_scale_fp32,
                    ),
                )
                p.daemon = True
                p.start()
                procs.append(p)

            any_error: Optional[str] = None
            for _ in range(len(procs)):
                _, errors, err_msg = q.get()
                if err_msg and any_error is None:
                    any_error = err_msg
                all_errors.extend(errors)

            for p in procs:
                p.join(timeout=60)
            for p in procs:
                if p.is_alive():
                    p.terminate()

            if any_error:
                raise RuntimeError(f"Worker error: {any_error}")

            all_errors = all_errors[:self.num_mc_samples]

        if len(all_errors) == 0:
            all_errors = [0.0]

        errs_tensor = torch.tensor(all_errors, dtype=torch.float32)
        qv = float(torch.quantile(errs_tensor, torch.tensor(self.quantile)))
        predicted = qv * self.safety_factor

        comp = self._estimate_components(
            x_cpu, w_cpu, num_samples=min(128, max(8, self.num_mc_samples // 8)),
            accum_dtype_eff=accum_dtype_eff,
        )

        return OracleResult(
            predicted_bound=predicted,
            quantile=self.quantile,
            safety_factor=self.safety_factor,
            sample_errors=all_errors,
            component_estimates=comp,
            meta={
                "num_samples": len(all_errors),
                "devices": self.devices,
                "strategy": str(self.strategy),
                "noise_mask": noise_mask,
                "accumulation_dim": x_cpu.shape[1],
                "accum_dtype_eff": str(accum_dtype_eff),
                "accum_k_cap": self.accum_k_cap,
                "gemm_params": dict(self.params),
            },
        )

    def _estimate_components(
        self,
        x_cpu: torch.Tensor,
        w_cpu: torch.Tensor,
        num_samples: int,
        accum_dtype_eff: torch.dtype,
    ) -> Dict[str, float]:
        def run(mask: Tuple[bool, bool, bool, bool]) -> float:
            device_id = self.devices[0] if len(self.devices) > 0 else None
            q = mp.Queue()
            self._worker_run(
                rank=0, device_id=device_id, x_cpu=x_cpu, w_cpu=w_cpu, params=self.params, strategy=self.strategy,
                num_local=num_samples, noise_mask=mask,
                seed_base=4321 if self.seeded else int(time.time()),
                return_queue=q,
                accum_dtype_eff=accum_dtype_eff, accum_k_cap=self.accum_k_cap,
                accum_sigma_same=self.accum_noise_scale_same, accum_sigma_fp32=self.accum_noise_scale_fp32,
            )
            _, errors, err_msg = q.get()
            if err_msg:
                raise RuntimeError(f"Component worker error: {err_msg}")
            if len(errors) == 0:
                return 0.0
            return float(torch.median(torch.tensor(errors, dtype=torch.float32)).item())

        return {
            "input_storage_error": run((True,  False, False, False)),
            "weight_storage_error": run((False, True,  False, False)),
            "accumulation_error":   run((False, False, True,  False)),
            "demote_error":         run((False, False, False, True )),
        }
