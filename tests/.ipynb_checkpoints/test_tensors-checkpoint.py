\

"""Tests for the tensor-network functions."""

import numpy as np
import pytest
import dmrg


def test_describe_scalar():
    information = dmrg.tensors.describe_tensor(5)

    assert information["type"] == "scalar"
    assert information["rank"] == 0
    assert information["notation"] == "c"
    assert information["dimensions"] == ()
    assert information["number_of_indices"] == 0
    assert information["number_of_elements"] == 1


def test_describe_vector():
    vector = np.array([1, 2, 3])

    information = dmrg.tensors.describe_tensor(vector)

    assert information["type"] == "vector"
    assert information["rank"] == 1
    assert information["notation"] == "v_i"
    assert information["dimensions"] == (3,)
    assert information["number_of_indices"] == 1
    assert information["number_of_elements"] == 3


def test_describe_matrix():
    matrix = np.zeros((2, 3))

    information = dmrg.tensors.describe_tensor(matrix)

    assert information["type"] == "matrix"
    assert information["rank"] == 2
    assert information["notation"] == "M_ij"
    assert information["dimensions"] == (2, 3)
    assert information["number_of_indices"] == 2
    assert information["number_of_elements"] == 6


def test_describe_higher_rank_tensor():
    tensor = np.zeros((2, 3, 4))

    information = dmrg.tensors.describe_tensor(tensor)

    assert information["type"] == "higher-rank tensor"
    assert information["rank"] == 3
    assert information["notation"] == "A with 3 indices"
    assert information["dimensions"] == (2, 3, 4)
    assert information["number_of_indices"] == 3
    assert information["number_of_elements"] == 24


def test_general_contraction():
    matrix = np.array([
        [1, 2],
        [3, 4],
    ])

    vector = np.array([1, 2])

    result = dmrg.tensors.contract(
        matrix,
        vector,
        axes=([1], [0]),
    )

    expected = np.array([5, 11])

    assert np.allclose(result, expected)


def test_matrix_vector_contraction():
    matrix = np.array([
        [1, 2],
        [3, 4],
    ])

    vector = np.array([1, 2])

    result = dmrg.tensors.matrix_vector_contraction(
        matrix,
        vector,
    )

    expected = np.array([5, 11])

    assert np.allclose(result, expected)


def test_matrix_vector_rejects_matrix_as_vector():
    matrix = np.eye(2)
    invalid_vector = np.eye(2)

    with pytest.raises(
        ValueError,
        match="vector must have rank 1",
    ):
        dmrg.tensors.matrix_vector_contraction(
            matrix,
            invalid_vector,
        )


def test_matrix_vector_rejects_incompatible_dimensions():
    matrix = np.zeros((2, 3))
    vector = np.zeros(2)

    with pytest.raises(
        ValueError,
        match="contracted dimensions must match",
    ):
        dmrg.tensors.matrix_vector_contraction(
            matrix,
            vector,
        )


def test_complete_contraction():
    left_vector = np.array([1, 2])

    matrix = np.array([
        [1, 2],
        [3, 4],
    ])

    right_vector = np.array([3, 4])

    result = dmrg.tensors.complete_contraction(
        left_vector,
        matrix,
        right_vector,
    )

    assert np.isclose(result, 61)


def test_three_tensor_contraction():
    A = np.ones((2, 3, 4))
    B = np.ones((3, 5))
    C = np.ones((4, 5, 6))

    result = dmrg.tensors.three_tensor_contraction(
        A,
        B,
        C,
    )

    assert result.shape == (2, 6)

    # Every result element contains 3 * 4 * 5 terms.
    assert np.allclose(result, 60)


def test_tensor_trace():
    matrix = np.array([
        [1, 2],
        [3, 4],
    ])

    result = dmrg.tensors.tensor_trace(matrix)

    assert np.isclose(result, 5)


def test_trace_rejects_non_square_matrix():
    matrix = np.zeros((2, 3))

    with pytest.raises(
        ValueError,
        match="matrix must be square",
    ):
        dmrg.tensors.tensor_trace(matrix)


def test_trace_rejects_vector():
    vector = np.array([1, 2])

    with pytest.raises(
        ValueError,
        match="rank-2 tensor",
    ):
        dmrg.tensors.tensor_trace(vector)


def test_outer_product():
    vector_a = np.array([1, 2])
    vector_b = np.array([3, 4])

    result = dmrg.tensors.outer_product(
        vector_a,
        vector_b,
    )

    expected = np.array([
        [3, 4],
        [6, 8],
    ])

    assert np.allclose(result, expected) 