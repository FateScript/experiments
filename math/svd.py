#!/usr/bin/env python3

# reference: https://www.youtube.com/watch?v=CpD9XlTu3ys

import numpy as np


def find_transform_matrix(vec1, vec2):
    """
    Find a matrix that transform two orthogonal vectors v1 and v2
    to another two orthogonal vectors.
    """
    assert np.allclose(vec1 @ vec2.T, 0)  # check if vec1 and vec2 are orthogonal
    diag1, diag2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    diag_mat = np.diag([1 / diag1, 1 / diag2])
    v_mat = np.array([vec1, vec2])
    matrix = (diag_mat @ v_mat).T  # default use U as np.diag(1, 1)

    # check if matrix works as expected
    u1, u2 = matrix @ vec1, matrix @ vec2
    assert np.allclose(u1 @ u2.T, 0)  # check if u1 and u2 are orthogonal
    print(f"Matrix:\n{matrix}\n could transform {vec1} and {vec2} to {u1} and {u2}")
    return matrix


def orthogonal_basis(matrix):
    u, s, v = np.linalg.svd(matrix)
    v1, v2 = v[0], v[1]  # original orthogonal basis
    assert np.allclose(v1 @ v2.T, 0)

    u1, u2 = u.T[0], u.T[1]  # orthogonal basis after apply matrix
    trans_u1, trans_u2 = matrix @ v1, matrix @ v2
    diag1, diag2 = np.linalg.norm(trans_u1), np.linalg.norm(trans_u2)
    assert s[0] == diag1 and s[1] == diag2
    norm_u1, norm_u2 = trans_u1 / diag1, trans_u2 / diag2

    assert np.allclose(norm_u1, u1) and np.allclose(norm_u2, u2)
    assert np.allclose(trans_u1 @ trans_u2.T, 0)  # check if u1 and u2 are orthogonal
    print(f"Orthogonal basis {v1} and {v2}")
    print(f"after apply matrix: \n{matrix}")
    print(f"become orthogonal basis {u1} and {u2}")


if __name__ == "__main__":
    matrix = np.array([[3, 4], [8, -6]])
    orthogonal_basis(matrix)

    print("\n")
    v1, v2 = np.array([5, 12]), np.array([24, -10])
    find_transform_matrix(v1, v2)
