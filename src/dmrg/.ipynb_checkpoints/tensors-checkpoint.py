

import numpy as np


def describe_tensor(tensor):
    """
    Describe a scalar, vector, matrix, or higher-rank tensor.

    Returns its type, rank, dimensions, number of indices,
    and number of elements.
    """
    tensor = np.asarray(tensor)
    rank = tensor.ndim

    if rank == 0:
        tensor_type = "scalar"
        notation = "c"

    elif rank == 1:
        tensor_type = "vector"
        notation = "v_i"

    elif rank == 2:
        tensor_type = "matrix"
        notation = "M_ij"

    else:
        tensor_type = "higher-rank tensor"
        notation = f"A with {rank} indices"

    return {
        "type": tensor_type,
        "rank": rank,
        "notation": notation,
        "dimensions": tensor.shape,
        "number_of_indices": rank,
        "number_of_elements": tensor.size,
    }

def contract(tensor_a, tensor_b, axes):
    """Contract two tensors over selected axes."""
    tensor_a = np.asarray(tensor_a)
    tensor_b = np.asarray(tensor_b)

    return np.tensordot(
        tensor_a,
        tensor_b,
        axes=axes,
    )


def matrix_vector_contraction(matrix, vector):
    """Calculate u_i = sum_j M_ij v_j."""
    matrix = np.asarray(matrix)
    vector = np.asarray(vector)

    if matrix.ndim != 2:
        raise ValueError("matrix must have rank 2.")

    if vector.ndim != 1:
        raise ValueError("vector must have rank 1.")

    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("The contracted dimensions must match.")

    return np.einsum("ij,j->i", matrix, vector)


def complete_contraction(left_vector, matrix, right_vector):
    """Calculate c = sum_ij u_i M_ij v_j."""
    return np.einsum(
        "i,ij,j->",
        np.asarray(left_vector),
        np.asarray(matrix),
        np.asarray(right_vector),
    )


def three_tensor_contraction(A, B, C):
    """Calculate T_im = sum_jkl A_ijk B_jl C_klm."""
    return np.einsum(
        "ijk,jl,klm->im",
        np.asarray(A),
        np.asarray(B),
        np.asarray(C),
    )


def tensor_trace(matrix):
    """Calculate Tr(M) = sum_i M_ii."""
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("The input must be a rank-2 tensor.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix must be square.")

    return np.trace(matrix)


def outer_product(vector_a, vector_b):
    """Calculate T_ij = a_i b_j."""
    return np.einsum(
        "i,j->ij",
        np.asarray(vector_a),
        np.asarray(vector_b),
    )