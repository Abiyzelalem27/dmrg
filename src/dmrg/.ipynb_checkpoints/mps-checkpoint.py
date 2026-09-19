

"""Basic Matrix Product State functions."""

import numpy as np


def state_to_mps(
    state,
    number_of_sites,
    physical_dimension=2,
    max_bond_dimension=None,
):
    """
    Convert a state vector into an open-boundary MPS using SVD.

    Each MPS tensor has shape:

        (left bond, physical index, right bond)

    For spin-1/2 systems, physical_dimension = 2.
    """
    state = np.asarray(state, dtype=complex).reshape(-1)

    expected_size = physical_dimension**number_of_sites

    if state.size != expected_size:
        raise ValueError(
            f"State must contain {expected_size} amplitudes."
        )

    norm = np.linalg.norm(state)

    if np.isclose(norm, 0.0):
        raise ValueError("The zero vector is not a quantum state.")

    remaining = state / norm
    tensors = []
    left_dimension = 1

    for _ in range(number_of_sites - 1):
        matrix = remaining.reshape(
            left_dimension * physical_dimension,
            -1,
        )

        U, singular_values, Vh = np.linalg.svd(
            matrix,
            full_matrices=False,
        )

        if max_bond_dimension is not None:
            kept = min(
                max_bond_dimension,
                singular_values.size,
            )

            U = U[:, :kept]
            singular_values = singular_values[:kept]
            Vh = Vh[:kept, :]

        right_dimension = singular_values.size

        tensor = U.reshape(
            left_dimension,
            physical_dimension,
            right_dimension,
        )

        tensors.append(tensor)

        remaining = singular_values[:, None] * Vh
        left_dimension = right_dimension

    final_tensor = remaining.reshape(
        left_dimension,
        physical_dimension,
        1,
    )

    tensors.append(final_tensor)

    return tensors


def mps_to_state(tensors):
    """Contract an open-boundary MPS into a full state vector."""
    state = np.asarray(tensors[0], dtype=complex)

    for tensor in tensors[1:]:
        state = np.tensordot(
            state,
            tensor,
            axes=(-1, 0),
        )

    return state.reshape(-1)


def bond_dimensions(tensors):
    """Return the internal MPS bond dimensions."""
    return [
        tensor.shape[2]
        for tensor in tensors[:-1]
    ]


def schmidt_coefficients(
    state,
    number_of_sites,
    cut,
    physical_dimension=2,
):
    """
    Calculate Schmidt coefficients across a selected cut.

    `cut` is the number of sites in subsystem A.
    """
    state = np.asarray(state, dtype=complex).reshape(-1)

    expected_size = physical_dimension**number_of_sites

    if state.size != expected_size:
        raise ValueError(
            f"State must contain {expected_size} amplitudes."
        )

    if not 1 <= cut < number_of_sites:
        raise ValueError(
            "cut must lie between two lattice sites."
        )

    state = state / np.linalg.norm(state)

    dimension_a = physical_dimension**cut
    dimension_b = physical_dimension ** (
        number_of_sites - cut
    )

    coefficient_matrix = state.reshape(
        dimension_a,
        dimension_b,
    )

    return np.linalg.svd(
        coefficient_matrix,
        compute_uv=False,
    )


def schmidt_rank(
    state,
    number_of_sites,
    cut,
    physical_dimension=2,
    tolerance=1e-12,
):
    """Return the Schmidt rank across a selected cut."""
    coefficients = schmidt_coefficients(
        state,
        number_of_sites,
        cut,
        physical_dimension,
    )

    return int(
        np.count_nonzero(coefficients > tolerance)
    )


def maximum_entanglement_entropy(bond_dimension, base=2):
    """
    Return the largest possible entropy for bond dimension D.

        S_max = log(D)
    """
    if bond_dimension < 1:
        raise ValueError(
            "bond_dimension must be positive."
        )

    return float(
        np.log(bond_dimension) / np.log(base)
    )


def mps_parameter_estimate(
    number_of_sites,
    physical_dimension,
    bond_dimension,
):
    """
    Estimate the number of MPS parameters.

        parameters approximately N * d * D^2
    """
    return (
        number_of_sites
        * physical_dimension
        * bond_dimension**2
    )

def maximum_entanglement_entropy(
    bond_dimension,
    base=2,
):
    """
    Maximum entropy supported by bond dimension D.

        S_max = log(D)
    """
    if bond_dimension < 1:
        raise ValueError(
            "bond_dimension must be positive."
        )

    return float(
        np.log(bond_dimension) / np.log(base)
    )


def mps_parameter_count(
    number_of_sites,
    physical_dimension,
    bond_dimension,
    periodic=False,
):
    """
    Count the number of MPS parameters.

    Open boundaries:
        2*d*D + (N-2)*d*D^2

    Periodic boundaries:
        N*d*D^2
    """
    N = number_of_sites
    d = physical_dimension
    D = bond_dimension

    if N < 2:
        raise ValueError(
            "number_of_sites must be at least 2."
        )

    if d < 1 or D < 1:
        raise ValueError(
            "Dimensions must be positive."
        )

    if periodic:
        return N * d * D**2

    return 2 * d * D + (N - 2) * d * D**2


def periodic_mps_amplitude(
    tensors,
    physical_indices,
):
    """
    Calculate one periodic-MPS coefficient.

        c_(i1,...,iN)
        = Tr(A[1,i1] A[2,i2] ... A[N,iN])

    Each tensor has shape (D, d, D).
    """
    if len(tensors) != len(physical_indices):
        raise ValueError(
            "One physical index is required per tensor."
        )

    matrices = []

    for tensor, physical_index in zip(
        tensors,
        physical_indices,
    ):
        tensor = np.asarray(tensor)

        if tensor.ndim != 3:
            raise ValueError(
                "Every MPS tensor must have rank 3."
            )

        if tensor.shape[0] != tensor.shape[2]:
            raise ValueError(
                "Periodic MPS tensors require equal left "
                "and right bond dimensions."
            )

        matrices.append(
            tensor[:, physical_index, :]
        )

    product = matrices[0]

    for matrix in matrices[1:]:
        product = product @ matrix

    return np.trace(product) 