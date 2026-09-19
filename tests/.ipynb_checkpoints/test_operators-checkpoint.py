

"""Tests for MPS operators and expectation values."""

import numpy as np
import pytest
import dmrg


def product_state_zero(number_of_sites):
    """
    Create the product-state MPS |00...0>.

    Every tensor has shape (1, 2, 1).
    """
    tensor = np.zeros((1, 2, 1), dtype=complex)
    tensor[0, 0, 0] = 1

    return [
        tensor.copy()
        for _ in range(number_of_sites)
    ]


def test_spin_operators():
    assert dmrg.operators.spin_x().shape == (2, 2)
    assert dmrg.operators.spin_y().shape == (2, 2)
    assert dmrg.operators.spin_z().shape == (2, 2)


def test_identity_operator():
    identity = dmrg.operators.identity_operator()

    assert identity.shape == (2, 2)
    assert np.allclose(identity, np.eye(2))


def test_identity_expectation_value():
    mps = product_state_zero(3)

    operators = [
        np.eye(2),
        np.eye(2),
        np.eye(2),
    ]

    result = dmrg.operators.expectation_value(
        mps,
        operators,
    )

    assert np.isclose(result, 1)


def test_local_spin_z():
    mps = product_state_zero(3)

    result = dmrg.operators.local_expectation_value(
        mps,
        dmrg.operators.spin_z(),
        site=1,
    )

    # S_z |0> = +(1/2)|0>
    assert np.isclose(result, 0.5)


def test_local_spin_x():
    mps = product_state_zero(3)

    result = dmrg.operators.local_expectation_value(
        mps,
        dmrg.operators.spin_x(),
        site=1,
    )

    assert np.isclose(result, 0)


def test_spin_correlation():
    mps = product_state_zero(3)

    result = dmrg.operators.spin_correlation(
        mps,
        first_site=0,
        second_site=2,
    )

    # <S_z(0) S_z(2)> = (1/2)(1/2) = 1/4
    assert np.isclose(result, 0.25)


def test_wrong_number_of_operators():
    mps = product_state_zero(3)

    operators = [
        np.eye(2),
        np.eye(2),
    ]

    with pytest.raises(
        ValueError,
        match="One local operator",
    ):
        dmrg.operators.expectation_value(
            mps,
            operators,
        )


def test_invalid_site():
    mps = product_state_zero(3)

    with pytest.raises(IndexError):
        dmrg.operators.local_expectation_value(
            mps,
            dmrg.operators.spin_z(),
            site=3,
        )


def test_same_correlation_sites_rejected():
    mps = product_state_zero(3)

    with pytest.raises(
        ValueError,
        match="different",
    ):
        dmrg.operators.spin_correlation(
            mps,
            first_site=1,
            second_site=1,
        )