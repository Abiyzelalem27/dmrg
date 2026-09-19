


import numpy as np
import pytest
import dmrg


def test_state_to_mps_two_qubit_product_state():
    """Test the conversion of |00> into an MPS."""
    state = np.array([1, 0, 0, 0], dtype=complex)

    tensors = dmrg.mps.state_to_mps(
        state,
        number_of_sites=2,
        max_bond_dimension=1,
    )

    assert len(tensors) == 2
    assert tensors[0].shape == (1, 2, 1)
    assert tensors[1].shape == (1, 2, 1)


def test_mps_to_state_product_state():
    """Check that converting to MPS and back recovers |00>."""
    original_state = np.array(
        [1, 0, 0, 0],
        dtype=complex,
    )

    tensors = dmrg.mps.state_to_mps(
        original_state,
        number_of_sites=2,
        max_bond_dimension=1,
    )

    reconstructed_state = dmrg.mps.mps_to_state(tensors)

    assert np.allclose(
        reconstructed_state,
        original_state,
    )


def test_bell_state_reconstruction():
    """Check that an MPS with D=2 represents a Bell state."""
    bell_state = np.array(
        [1, 0, 0, 1],
        dtype=complex,
    ) / np.sqrt(2)

    tensors = dmrg.mps.state_to_mps(
        bell_state,
        number_of_sites=2,
        max_bond_dimension=2,
    )

    reconstructed_state = dmrg.mps.mps_to_state(tensors)

    assert np.allclose(
        reconstructed_state,
        bell_state,
    )


def test_bond_dimensions():
    """Check the bond dimension of a Bell-state MPS."""
    bell_state = np.array(
        [1, 0, 0, 1],
        dtype=complex,
    ) / np.sqrt(2)

    tensors = dmrg.mps.state_to_mps(
        bell_state,
        number_of_sites=2,
        max_bond_dimension=2,
    )

    dimensions = dmrg.mps.bond_dimensions(tensors)

    assert dimensions == [2]


def test_product_state_schmidt_coefficients():
    """A product state has one nonzero Schmidt coefficient."""
    product_state = np.array(
        [1, 0, 0, 0],
        dtype=complex,
    )

    coefficients = dmrg.mps.schmidt_coefficients(
        product_state,
        number_of_sites=2,
        cut=1,
    )

    expected = np.array([1, 0])

    assert np.allclose(coefficients, expected)


def test_bell_state_schmidt_coefficients():
    """A Bell state has two equal Schmidt coefficients."""
    bell_state = np.array(
        [1, 0, 0, 1],
        dtype=complex,
    ) / np.sqrt(2)

    coefficients = dmrg.mps.schmidt_coefficients(
        bell_state,
        number_of_sites=2,
        cut=1,
    )

    expected = np.array([
        1 / np.sqrt(2),
        1 / np.sqrt(2),
    ])

    assert np.allclose(coefficients, expected)


def test_product_state_schmidt_rank():
    """A product state has Schmidt rank one."""
    product_state = np.array(
        [1, 0, 0, 0],
        dtype=complex,
    )

    rank = dmrg.mps.schmidt_rank(
        product_state,
        number_of_sites=2,
        cut=1,
    )

    assert rank == 1


def test_bell_state_schmidt_rank():
    """An entangled Bell state has Schmidt rank two."""
    bell_state = np.array(
        [1, 0, 0, 1],
        dtype=complex,
    ) / np.sqrt(2)

    rank = dmrg.mps.schmidt_rank(
        bell_state,
        number_of_sites=2,
        cut=1,
    )

    assert rank == 2


def test_maximum_entanglement_entropy():
    """
    For D=4 and logarithm base 2:

        S_max = log_2(4) = 2.
    """
    entropy = dmrg.mps.maximum_entanglement_entropy(
        bond_dimension=4,
        base=2,
    )

    assert np.isclose(entropy, 2)


def test_mps_parameter_estimate():
    """
    Test the estimate:

        N * d * D^2.
    """
    parameters = dmrg.mps.mps_parameter_estimate(
        number_of_sites=5,
        physical_dimension=2,
        bond_dimension=4,
    )

    assert parameters == 160


def test_invalid_state_size():
    """Reject a vector with the wrong Hilbert-space dimension."""
    invalid_state = np.array([1, 0, 0])

    with pytest.raises(
        ValueError,
        match="State must contain",
    ):
        dmrg.mps.state_to_mps(
            invalid_state,
            number_of_sites=2,
        )


def test_zero_state_is_invalid():
    """The zero vector is not a quantum state."""
    zero_state = np.zeros(4)

    with pytest.raises(
        ValueError,
        match="zero vector",
    ):
        dmrg.mps.state_to_mps(
            zero_state,
            number_of_sites=2,
        )


def test_invalid_cut():
    """The cut must lie between lattice sites."""
    state = np.array([1, 0, 0, 0])

    with pytest.raises(
        ValueError,
        match="cut must lie",
    ):
        dmrg.mps.schmidt_coefficients(
            state,
            number_of_sites=2,
            cut=2,
        )


def test_invalid_bond_dimension_for_entropy():
    """The bond dimension must be positive."""
    with pytest.raises(
        ValueError,
        match="bond_dimension must be positive",
    ):
        dmrg.mps.maximum_entanglement_entropy(0)

def test_maximum_entanglement_entropy():
    result = dmrg.mps.maximum_entanglement_entropy(
        bond_dimension=4,
        base=2,
    )

    assert np.isclose(result, 2)


def test_open_mps_parameter_count():
    result = dmrg.mps.mps_parameter_count(
        number_of_sites=5,
        physical_dimension=2,
        bond_dimension=4,
        periodic=False,
    )

    assert result == 112


def test_periodic_mps_parameter_count():
    result = dmrg.mps.mps_parameter_count(
        number_of_sites=5,
        physical_dimension=2,
        bond_dimension=4,
        periodic=True,
    )

    assert result == 160


def test_periodic_mps_amplitude():
    # A D=1 periodic MPS representing |00>.
    tensor = np.array([
        [[1], [0]],
    ])

    amplitude_00 = dmrg.mps.periodic_mps_amplitude(
        [tensor, tensor],
        [0, 0],
    )

    amplitude_01 = dmrg.mps.periodic_mps_amplitude(
        [tensor, tensor],
        [0, 1],
    )

    assert np.isclose(amplitude_00, 1)
    assert np.isclose(amplitude_01, 0) 