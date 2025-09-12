#!/usr/bin/env python3

import matplotlib.pyplot as plt


def decode_mxfp8_e4m3(byte_val):
    sign = (byte_val >> 7) & 0x1
    exponent = (byte_val >> 3) & 0xF  # 4 bit
    mantissa = byte_val & 0x7  # 3 bit
    exp_bias = 7  # 2^(4-1) - 1

    if exponent == 0:
        significand = mantissa / 8.0  # mantissa / 2^3
        value = significand * (2 ** (1 - exp_bias))
    elif exponent == 0xF and mantissa == 7:
        # inf for e4m3 fp8, nan for other formats
        value = float("inf")
    else:  # normal number
        actual_exponent = exponent - exp_bias
        significand = 1.0 + mantissa / 8.0
        value = significand * (2**actual_exponent)

    return -value if sign == 1 else value


def decode_float_number(byte_val, exp_bit=5, mantisa_bit=2) -> float:
    if exp_bit == 4 and mantisa_bit == 3:  # e4m3 mxfp8
        return decode_mxfp8_e4m3(byte_val)

    exp_mask = (1 << exp_bit) - 1
    mantisa_mask = (1 << mantisa_bit) - 1
    exp_bias = (1 << (exp_bit - 1)) - 1

    sign = (byte_val >> (exp_bit + mantisa_bit)) & 0x1
    exponent = (byte_val >> mantisa_bit) & exp_mask
    mantissa = byte_val & mantisa_mask

    if exponent == 0:  # subnormal
        significand = mantissa / (1 << mantisa_bit)
        value = significand * (2 ** (1 - exp_bias))
    elif exponent == exp_mask:  # all 1s in exponent, inf or nan
        if mantissa == mantisa_mask:
            value = float("inf")
        else:
            return float("nan")
    else:  # normal number
        actual_exponent = exponent - exp_bias
        significand = 1.0 + mantissa / (1 << mantisa_bit)
        value = significand * (2**actual_exponent)

    return -value if sign else value


def print_fp8_values():
    print("=" * 60)
    print(f"{'Bin':<10} {'Hex':<6} {'Dec':<6} {'e4m3':<15} {'e5m2':<15}")
    print("-" * 60)

    for i in range(256):
        binary = format(i, "08b")
        hex_val = format(i, "02X")
        e4m3_val = decode_float_number(i, exp_bit=4, mantisa_bit=3)
        e5m2_val = decode_float_number(i, exp_bit=5, mantisa_bit=2)

        print(f"{binary:<10} {hex_val:<6} {i:<6} {e4m3_val:<15} {e5m2_val:<15}")

    print("\n" + "=" * 60)


def plot_fp8_values(max_value=10):
    e4m3_values = []
    e5m2_values = []
    indices = []

    for i in range(256):
        e4m3_val = decode_float_number(i, exp_bit=4, mantisa_bit=3)
        e5m2_val = decode_float_number(i, exp_bit=5, mantisa_bit=2)

        if abs(e4m3_val) > max_value:
            break

        e4m3_values.append(e4m3_val)
        e5m2_values.append(e5m2_val)
        indices.append(i)

    plt.figure(figsize=(10, 6))
    plt.plot(indices, e4m3_values, label="e4m3 Values", color='blue', linestyle='-', marker='o')
    plt.plot(indices, e5m2_values, label="e5m2 Values", color='red', linestyle='-', marker='x')

    plt.title("FP8 Value Comparison: e4m3 vs e5m2")
    plt.xlabel("i (Index)")
    plt.ylabel("Decoded Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fp8_distribution(max_value=10):
    import seaborn as sns
    e4m3_values = []
    e5m2_values = []

    for i in range(256):
        e4m3_val = decode_float_number(i, exp_bit=4, mantisa_bit=3)
        e5m2_val = decode_float_number(i, exp_bit=5, mantisa_bit=2)
        if abs(e4m3_val) > max_value:
            break

        e4m3_values.append(e4m3_val)
        e5m2_values.append(e5m2_val)

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(e4m3_values, kde=True, color='blue', label="e4m3 Values", stat="density", bins=30)
    sns.histplot(e5m2_values, kde=True, color='red', label="e5m2 Values", stat="density", bins=30)

    plt.title("Distribution of FP8 Values: e4m3 vs e5m2")
    plt.xlabel("Decoded Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print_fp8_values()
    # e4m3 == e5m2 == 2.0 when i = 64
    # plot_fp8_values(max_value=3)
    # plot_fp8_distribution(max_value=3)
