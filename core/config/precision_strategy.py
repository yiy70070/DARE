# precision_estimation/core/config/precision_strategy.py
"""
Precision strategy utilities for precision_estimation project.

Provides:
- PrecisionStrategy dataclass
- get_precision_strategy(name)
- ulp_like(x, dtype) : returns element-wise ULP estimate with same shape as x (robust across all devices/dtypes)
- quantize_to_dtype(x, dtype) : simulates tensor quantization to given dtype (with IEEE round-to-nearest)
- promote_exact(x, to_dtype) : exact promotion (low->high precision exact conversion)
- demote_with_round(x, to_dtype) : simulates precision demotion and returns demoted value (with rounding error estimate)

Implementation considerations:
- torch.nextafter is unavailable for Half on CUDA in some PyTorch versions; use fallback for compatibility.
- For float16/bfloat16 etc., use relative eps-based estimation: ulp(x) ≈ |x| * eps(dtype)
  and use dtype.tiny (smallest subnormal value) as lower bound for values near 0 to ensure numerical stability.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

import torch



# -------------------------
# Precision strategy type
# -------------------------
@dataclass
class PrecisionStrategy:
    name: str
    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    compute_dtype: torch.dtype
    output_dtype: torch.dtype
    input_quant: str = "none"
    weight_quant: str = "none"
    output_quant: str = "none"
    quant_params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            "PrecisionStrategy("
            f"name='{self.name}', "
            f"input_dtype={self.input_dtype}, "
            f"weight_dtype={self.weight_dtype}, "
            f"compute_dtype={self.compute_dtype}, "
            f"output_dtype={self.output_dtype}, "
            f"input_quant='{self.input_quant}', "
            f"weight_quant='{self.weight_quant}', "
            f"output_quant='{self.output_quant}', "
            f"quant_params={self.quant_params}"
            ")"
        )


# -------------------------
# Common strategy factory
# -------------------------
def get_precision_strategy(name: str) -> PrecisionStrategy:
    """
    Returns common mixed-precision strategies.
    Supported names:
      - 'FP32'                    : all FP32
      - 'FP16_all'                : all FP16
      - 'FP16_compute_FP32_accum' : FP16 compute, FP32 accumulation (example)
      - 'FP16_input_FP32_weight_FP32_compute_accum' : your default experimental setting
      - 'BF16_compute'            : BF16 compute
      - 'INT8_W8A8_PT'            : int8 symmetric per-tensor fake-quant for both input/weight, FP32 compute/output
      - 'INT8_W8PC_A8PT'          : int8 input per-tensor + weight per-channel(fake-quant), FP32 compute/output
      - 'INT8_W8PC_A8CLIP'        : weight per-channel + activation per-tensor with percentile clip, FP32 compute/output
    """
    name = name.strip()
    mapping = {
        'FP32': PrecisionStrategy('FP32', torch.float32, torch.float32, torch.float32, torch.float32),
        'FP16_all': PrecisionStrategy('FP16_all', torch.float16, torch.float16, torch.float16, torch.float16),
        'FP16_compute_FP32_accum': PrecisionStrategy('FP16_compute_FP32_accum', torch.float16, torch.float16, torch.float32, torch.float16),
        # Your default experiment (input FP16, weight FP32, compute FP32, output FP16)
        'FP16_input_FP32_weight_FP32_compute_accum': PrecisionStrategy(
            'FP16_input_FP32_weight_FP32_compute_accum',
            torch.float16, torch.float32, torch.float32, torch.float16
        ),
        'BF16_compute': PrecisionStrategy('BF16_compute', torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16),
        'INT8_W8A8_PT': PrecisionStrategy(
            'INT8_W8A8_PT',
            torch.float32, torch.float32, torch.float32, torch.float32,
            input_quant="int8_affine_pt",
            weight_quant="int8_affine_pt",
            output_quant="none",
            quant_params={},
        ),
        'INT8_W8PC_A8PT': PrecisionStrategy(
            'INT8_W8PC_A8PT',
            torch.float32, torch.float32, torch.float32, torch.float32,
            input_quant="int8_affine_pt",
            weight_quant="int8_affine_pc",
            output_quant="none",
            quant_params={"axis": 0},
        ),
        'INT8_W8PC_A8CLIP': PrecisionStrategy(
            'INT8_W8PC_A8CLIP',
            torch.float32, torch.float32, torch.float32, torch.float32,
            input_quant="int8_affine_pt_clip",
            weight_quant="int8_affine_pc",
            output_quant="none",
            quant_params={"axis": 0, "clip_percentile": 99.9},
        ),
    }

    if name in mapping:
        return mapping[name]
    else:
        raise ValueError(f"Unknown precision strategy {name}")


# -------------------------
# ULP estimation function (robust implementation)
# -------------------------
def _finfo_for_dtype(dtype: torch.dtype) -> torch.finfo:
    # Raise error for non-float types
    if dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError(f"ulp_like only supports float dtypes, got {dtype}")
    return torch.finfo(dtype)


def ulp_like(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns element-wise ULP estimate (tensor) with same shape as x.
    Implementation strategy:
      1) For dtypes that can safely call torch.nextafter (float32/64 available on mainstream cuda/cpu versions),
         try using nextafter to compute more accurate differences
      2) If nextafter is unavailable for the dtype on current device (e.g. float16 on some CUDA versions),
         use fallback: ulp(x) ≈ max( |x| * eps(dtype), tiny(dtype) )
      3) For positions where x is 0, return tiny(dtype) (smallest subnormal absolute value) to avoid 0
    Note: Return value is on x.device (if x is on GPU), caller doesn't need additional .to(device)
    """
    if not torch.is_tensor(x):
        raise ValueError("ulp_like requires a tensor x")

    target_device = x.device
    finfo = _finfo_for_dtype(dtype)
    eps = finfo.eps
    tiny = finfo.tiny

    # working tensor on same device/shape for broadcasting
    x_tgt = x.to(dtype=torch.float32) if dtype == torch.float16 or dtype == torch.bfloat16 else x.to(dtype=dtype)

    # try to use nextafter when supported for the dtype on that device
    try:
        # nextafter needs both operands same dtype; create +inf tensor in that dtype/device
        if dtype in (torch.float32, torch.float64):
            x_in_dtype = x.to(dtype=dtype)
            inf_tensor = torch.full_like(x_in_dtype, float('inf'), dtype=dtype, device=target_device)
            # torch.nextafter may throw if unsupported; wrap in try
            next_vals = torch.nextafter(x_in_dtype, inf_tensor)
            ulp = (next_vals - x_in_dtype).abs()
            # protect extremely small values: ensure at least tiny
            ulp = torch.clamp(ulp, min=torch.tensor(tiny, device=target_device, dtype=ulp.dtype))
            return ulp.to(device=target_device)
        else:
            # float16 / bfloat16: some platforms / cuda versions don't implement nextafter_cuda for half
            # fallback to relative eps estimate
            raise RuntimeError("fallback to relative estimation for half/bfloat")
    except Exception:
        # fallback: relative eps * |x| with floor at tiny
        # Use float32 arithmetic for stability then cast to target device/dtype
        with torch.no_grad():
            abs_x = x.abs().to(dtype=torch.float32, device=target_device)
            ulp_est = abs_x * float(eps)  # relative measure
            # values near zero: use tiny
            tiny_tensor = torch.full_like(ulp_est, float(tiny), device=target_device, dtype=ulp_est.dtype)
            ulp_est = torch.maximum(ulp_est, tiny_tensor)
            # cast to requested dtype if needed (we return float32 tensor for numerical operations)
            return ulp_est.to(device=target_device)


# -------------------------
# Quantization / promotion / demotion simulation tools
# -------------------------
def quantize_to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Quantize tensor x to dtype and return to float32 (simulates storing to that dtype then reading back to compute width).
    For example: quantizing FP32->FP16 then returning FP32 represents that storage error has been introduced 
    but subsequent computation proceeds in FP32.
    Uses PyTorch's dtype cast to approximate IEEE rounding.
    """
    if dtype == x.dtype:
        return x.clone()
    # cast to dtype then back to float32 for further processing
    quant = x.to(dtype=dtype)
    # return in float32 for consistent downstream arithmetic
    return quant.to(dtype=torch.float32)


def promote_exact(x: torch.Tensor, to_dtype: torch.dtype) -> torch.Tensor:
    """
    Exact promotion (low->high): low precision to high precision conversion is exact per IEEE (no additional error).
    Therefore direct cast is sufficient.
    """
    return x.to(dtype=to_dtype)


def demote_with_round(x: torch.Tensor, to_dtype: torch.dtype) -> torch.Tensor:
    """
    Simulate precision demotion (e.g. FP32 -> FP16) and return demoted tensor (with rounding error).
    Uses PyTorch's cast (default rounding) directly, and returns float32 for subsequent comparison.
    """
    dem = x.to(dtype=to_dtype)
    return dem.to(dtype=torch.float32)


# -------------------------
# INT8 fake-quant helpers
# -------------------------
def clip_by_percentile(x: torch.Tensor, percentile: float) -> torch.Tensor:
    """
    Symmetric percentile clip for activations. Uses |x| distribution.
    Percentile in [0,100]; minimum threshold clamped to 1e-8 to avoid divide-by-zero.
    """
    if not torch.is_tensor(x):
        raise ValueError("clip_by_percentile expects a tensor input")
    if x.numel() == 0:
        return x.to(dtype=torch.float32)

    q = max(0.0, min(100.0, float(percentile))) / 100.0
    # torch.quantile handles GPU/CPU
    thr = torch.quantile(x.abs().flatten(), torch.tensor(q, device=x.device, dtype=x.dtype))
    thr = torch.clamp(thr, min=torch.tensor(1e-8, device=x.device, dtype=thr.dtype))
    return x.clamp(-thr, thr)


def quantize_int8_affine_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric affine per-tensor int8 fake-quant: quantize to int8 then dequantize back to float32.
    """
    if not torch.is_tensor(x):
        raise ValueError("quantize_int8_affine_per_tensor expects a tensor input")
    if x.numel() == 0:
        return x.to(dtype=torch.float32)

    x_f = x.to(dtype=torch.float32)
    max_abs = x_f.abs().max()
    scale = torch.clamp(max_abs / 127.0, min=torch.tensor(1e-8, device=x.device, dtype=torch.float32))
    q = torch.clamp(torch.round(x_f / scale), -127, 127)
    dq = q * scale
    return dq.to(dtype=torch.float32)


def quantize_int8_affine_per_channel(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """
    Symmetric affine per-channel int8 fake-quant along given axis (keepdim).
    axis default 0 for Conv2D out_channel.
    """
    if not torch.is_tensor(x):
        raise ValueError("quantize_int8_affine_per_channel expects a tensor input")
    if x.numel() == 0:
        return x.to(dtype=torch.float32)

    axis = int(axis)
    if axis < 0:
        axis = x.dim() + axis
    if axis < 0 or axis >= x.dim():
        raise ValueError(f"axis {axis} out of range for tensor dim {x.dim()}")

    x_f = x.to(dtype=torch.float32)
    max_abs = x_f.abs().amax(dim=axis, keepdim=True)
    scale = torch.clamp(max_abs / 127.0, min=torch.tensor(1e-8, device=x.device, dtype=torch.float32))
    q = torch.clamp(torch.round(x_f / scale), -127, 127)
    dq = q * scale
    return dq.to(dtype=torch.float32)


def apply_input_quant(x: torch.Tensor, s: PrecisionStrategy) -> torch.Tensor:
    """
    Unified input fake-quant entry.
    """
    if s.input_quant == "none":
        return quantize_to_dtype(x, s.input_dtype).to(dtype=torch.float32)
    if s.input_quant == "int8_affine_pt":
        return quantize_int8_affine_per_tensor(x)
    if s.input_quant == "int8_affine_pt_clip":
        percentile = (s.quant_params or {}).get("clip_percentile", 99.9)
        x_clip = clip_by_percentile(x, percentile)
        return quantize_int8_affine_per_tensor(x_clip)
    raise ValueError(f"Unsupported input_quant: {s.input_quant}")


def apply_weight_quant(w: torch.Tensor, s: PrecisionStrategy) -> torch.Tensor:
    """
    Unified weight fake-quant entry.
    """
    if s.weight_quant == "none":
        return quantize_to_dtype(w, s.weight_dtype).to(dtype=torch.float32)
    if s.weight_quant == "int8_affine_pt":
        return quantize_int8_affine_per_tensor(w)
    if s.weight_quant == "int8_affine_pc":
        axis = (s.quant_params or {}).get("axis", 0)
        return quantize_int8_affine_per_channel(w, axis=axis)
    raise ValueError(f"Unsupported weight_quant: {s.weight_quant}")


def apply_output_quant(y: torch.Tensor, s: PrecisionStrategy) -> torch.Tensor:
    """
    Unified output fake-quant entry. Currently uses demote_with_round for non-quantized outputs.
    """
    if s.output_quant == "none":
        return demote_with_round(y, s.output_dtype)
    if s.output_quant == "int8_affine_pt":
        return quantize_int8_affine_per_tensor(y)
    if s.output_quant == "int8_affine_pc":
        axis = (s.quant_params or {}).get("axis", 0)
        return quantize_int8_affine_per_channel(y, axis=axis)
    raise ValueError(f"Unsupported output_quant: {s.output_quant}")


# -------------------------
# Other convenience functions (examples)
# -------------------------
def ulp_scalar(value: float, dtype: torch.dtype) -> float:
    """
    Estimate ULP for a single float value (for small examples/testing).
    """
    finfo = _finfo_for_dtype(dtype)
    if value == 0.0:
        return float(finfo.tiny)
    else:
        return max(abs(value) * finfo.eps, float(finfo.tiny))