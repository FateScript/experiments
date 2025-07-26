#!/usr/bin/env python3

import numpy as np


def qrange(num_bits=8, signed=True):
    if signed:
        # int8 example: [-128, 127]
        exp = num_bits - 1
        qmin, qmax = -(1 << exp), (1 << exp) - 1
    else:
        # uint8 example: [0, 255]
        qmin, qmax = 0, (1 << num_bits) - 1
    return qmin, qmax


def choose_scale_zero_point(min_val, max_val, num_bits=8, signed=True, symmetric=False):
    """
    Calculates the scale and zero_point for linear quantization.

    Args:
        min_val (float): Minimum value of the data to be quantized.
        max_val (float): Maximum value of the data to be quantized.
        num_bits (int, optional): Number of bits for quantization. Defaults to 8.
        signed (bool, optional): Whether to use signed quantization. Defaults to True.
        symmetric (bool, optional): Whether to use symmetric quantization.
            If True, zero_point is set to 0 (128 for uint8), and the range is symmetric around zero.
            If False, zero_point is computed from min/max. Defaults to False.

    Returns:
        Tuple[float, int]: The scale and zero_point for quantization.
    """
    qmin, qmax = qrange(num_bits, signed)

    if symmetric:
        max_abs = max(abs(min_val), abs(max_val))
        scale = max_abs / qmax if max_abs != 0 else 1.0
        zero_point = 0 if signed else (qmax + qmin) // 2
    else:
        # avoid div 0 when max_val == min_val
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            # zero_point is value that map zero to the quantized range.
            # zero = (0 - min_val) / scale + offset (offset is qmin)
            zero_point = np.round(qmin - min_val / scale).astype(np.int32)
            zero_point = np.clip(zero_point, qmin, qmax)

    return float(scale), int(zero_point)


def quantize(x, scale, zero_point, num_bits=8, signed=True):
    """float -> int quantization
    Equation: q = round((x - 0) / scale + offset), where offset is zero_point.
    which is: q = round(x / scale + zero_point)
    """
    qmin, qmax = qrange(num_bits, signed)
    q = np.round(x / scale + zero_point).astype(np.int32)
    q = np.clip(q, qmin, qmax).astype(np.int32)
    return q


def dequantize(q, scale, zero_point):
    """int -> float dequantization"""
    return (q.astype(np.float32) - zero_point) * scale


def weight_act_quant():
    np.random.seed(0)
    x = np.random.randn(1000).astype(np.float32) * 2.5

    # Non-symmetric quantization (common for activations)
    scale_a, zp_a = choose_scale_zero_point(x.min(), x.max(), num_bits=8, signed=True, symmetric=False)
    x_int = quantize(x, scale_a, zp_a, num_bits=8, signed=True)
    x_hat = dequantize(x_int, scale_a, zp_a)

    mse = np.mean((x - x_hat) ** 2)
    max_abs_err = np.max(np.abs(x - x_hat))

    print("==== Activation (non-symmetric) quant ====")
    print(f"scale={scale_a:.6f}, zero_point={zp_a}")
    print(f"MSE={mse:.6f}, MaxAbsErr={max_abs_err:.6f}\n")

    # Symmetric quantization (common for weights)
    scale_s, zp_s = choose_scale_zero_point(x.min(), x.max(), num_bits=8, signed=True, symmetric=True)
    x_int_s = quantize(x, scale_s, zp_s, num_bits=8, signed=True)
    x_hat_s = dequantize(x_int_s, scale_s, zp_s)

    mse_s = np.mean((x - x_hat_s) ** 2)
    max_abs_err_s = np.max(np.abs(x - x_hat_s))

    print("==== Weight (symmetric) quant ====")
    print(f"scale={scale_s:.6f}, zero_point={zp_s}")
    print(f"MSE={mse_s:.6f}, MaxAbsErr={max_abs_err_s:.6f}\n")


def linear_quant_inference():
    # y = x @ W^T (ignore bias for simplicity)
    np.random.seed(0)
    batch, in_f, out_f = 4, 16, 8
    X = np.random.randn(batch, in_f).astype(np.float32)
    W = np.random.randn(out_f, in_f).astype(np.float32) * 0.5

    signed = True
    # activation quant (non symmetric)
    # activation is usually non-symmetric, e.g. ReLU output
    sx, zpx = choose_scale_zero_point(X.min(), X.max(), symmetric=False, signed=signed)
    X_q = quantize(X, sx, zpx, signed=signed)

    # weight quant (symmetric)
    # distribution of W is usually supposed to be symmetric
    sw, zpw = choose_scale_zero_point(W.min(), W.max(), symmetric=True, signed=signed)
    W_q = quantize(W, sw, zpw, signed=signed)

    # ground truth value of Y
    Y_float = X @ W.T

    # Quantized inference equation:
    # Y = x @ W^T
    #   ≈ ( (X_q - zpx) * sx ) @ ( (W_q - zpw) * sw )^T
    #   = sx * sw * (X_q - zpx) @ (W_q - zpw)^T
    Y_int32 = (X_q - zpx) @ (W_q - zpw).T  # int32 calculation
    Y_hat = (sx * sw) * Y_int32.astype(np.float32)

    mse_linear = np.mean((Y_float - Y_hat) ** 2)
    max_abs_err_linear = np.max(np.abs(Y_float - Y_hat))

    print("==== Linear inference（int32 calculate and dequantize） ====")
    print(f"Signed input: {signed}")
    print(f"scale_x={sx:.6f}, zero_point_x={zpx}, scale_w={sw:.6f}, zero_point_w={zpw}")
    print(f"MSE={mse_linear:.6f}, MaxAbsErr={max_abs_err_linear:.6f}")


if __name__ == "__main__":
    weight_act_quant()
    linear_quant_inference()
