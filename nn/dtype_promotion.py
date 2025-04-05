#!/usr/bin/env python3

import itertools
from collections import defaultdict

import numpy as np
import torch


def data_op(a, b, operation: str):
    if operation == '+':
        result = a + b
    elif operation == '-':
        result = a - b
    elif operation == '*':
        result = a * b
    elif operation == '/':
        result = a / b
    else:
        raise ValueError("Unsupported operation")
    return result


def check_dtype_cast(dtype1, dtype2, operation: str, backend: str = "torch"):
    if backend == "torch":
        a = torch.tensor(1.0, dtype=dtype1)
        b = torch.tensor(1.0, dtype=dtype2)
    elif backend == "numpy":
        a = np.array([1.0], dtype=dtype1)
        b = np.array([1.0], dtype=dtype2)

    result = data_op(a, b, operation=operation)
    return result.dtype


def check_zero_dim_type(dtype, scalar, operation: str, backend: str = "torch"):
    if backend == "torch":
        a = torch.tensor(1.0, dtype=dtype)
    elif backend == "numpy":
        a = np.array(1.0, dtype=dtype)

    result = data_op(a, scalar, operation=operation)
    return result.dtype


def zero_dim_dtype_table(backend: str = "torch"):
    assert backend in ["torch", "numpy"]
    scalars = [10, 10.2]
    if backend == "torch":
        dtypes = [
            torch.float16, torch.float32, torch.float64, torch.bfloat16,
            torch.int8, torch.int16, torch.int32, torch.int64,
        ]
    elif backend == "numpy":
        dtypes = [
            np.float16, np.float32, np.float64,
            np.int8, np.int16, np.int32, np.int64,
        ]

    for dtype in dtypes:
        for scalar in scalars:
            scalar_type = "int" if isinstance(scalar, int) else "float"
            result_dict = defaultdict(list)
            for op in ["+", "-", "*", "/"]:
                result_dtype = check_zero_dim_type(dtype, scalar, op, backend=backend)
                result_dict[result_dtype].append(op)

            for result_dtype, ops in result_dict.items():
                op_str = "".join(ops)
                print(f"{dtype} {op_str} {scalar_type} -> {result_dtype}")
        print()


def dtype_cast_table(backend: str = "torch"):
    assert backend in ["torch", "numpy"]
    if backend == "torch":
        dtypes = [
            torch.float16, torch.float32, torch.float64, torch.bfloat16,
            torch.int8, torch.int16, torch.int32, torch.int64,
        ]
    elif backend == "numpy":
        dtypes = [
            np.float16, np.float32, np.float64,
            np.int8, np.int16, np.int32, np.int64,
        ]

    for dtype1, dtype2 in itertools.combinations_with_replacement(dtypes, 2):
        result_dict = defaultdict(list)
        for op in ["+", "-", "*", "/"]:
            result_dtype = check_dtype_cast(dtype1, dtype2, op, backend=backend)
            result_dict[result_dtype].append(op)

        for result_dtype, ops in result_dict.items():
            op_str = "".join(ops)
            print(f"{dtype1} {op_str} {dtype2} -> {result_dtype}")
        print()


if __name__ == "__main__":
    backends = ["torch", "numpy"]
    for backend in backends:
        print(f"backend: {backend}")
        dtype_cast_table(backend=backend)
        print("---" * 30)

    print("\nScalar(Zero-dim) Type\n" + "---" * 30)
    for backend in backends:
        print(f"backend: {backend}")
        zero_dim_dtype_table(backend=backend)
        print("---" * 30)
