



import numpy as np


def expectation_value(mps, operators):
    """
    Calculate

        <psi| O_1 O_2 ... O_N |psi>
        --------------------------------
                  <psi|psi>

    without constructing the complete wave function.

    Each MPS tensor has shape:

        (left bond, physical dimension, right bond)
    """
    if len(mps) != len(operators):
        raise ValueError(
            "One local operator is required for every MPS site."
        )

    # The left boundary of an open-boundary MPS has dimension 1.
    environment = np.ones((1, 1), dtype=complex)

    for tensor, operator in zip(mps, operators):
        tensor = np.asarray(tensor)
        operator = np.asarray(operator)

        if tensor.ndim != 3:
            raise ValueError("Every MPS tensor must have rank 3.")

        if operator.ndim != 2:
            raise ValueError("Every local operator must be a matrix.")

        physical_dimension = tensor.shape[1]

        if operator.shape != (
            physical_dimension,
            physical_dimension,
        ):
            raise ValueError(
                "Operator and physical dimensions must match."
            )

        environment = np.einsum(
            "xy,xpb,pq,yqc->bc",
            environment,
            tensor.conj(),
            operator,
            tensor,
        )

    numerator = environment[0, 0]

    # Calculate <psi|psi> for normalization.
    norm_environment = np.ones((1, 1), dtype=complex)

    for tensor in mps:
        tensor = np.asarray(tensor)

        norm_environment = np.einsum(
            "xy,xpb,y pc->bc".replace(" ", ""),
            norm_environment,
            tensor.conj(),
            tensor,
        )

    denominator = norm_environment[0, 0]

    if np.isclose(denominator, 0):
        raise ValueError("The MPS has zero norm.")

    result = numerator / denominator

    # Convert numerically real results to ordinary real numbers.
    return np.real_if_close(result)


def identity_operator():
    """Return the identity operator for one spin-1/2 site."""
    return np.eye(2, dtype=complex)


def spin_x():
    """Return the spin-1/2 operator S_x."""
    return 0.5 * np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )


def spin_y():
    """Return the spin-1/2 operator S_y."""
    return 0.5 * np.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=complex,
    )


def spin_z():
    """Return the spin-1/2 operator S_z."""
    return 0.5 * np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=complex,
    )


def local_expectation_value(mps, operator, site):
    """
    Calculate the expectation value of an operator at one site.

        <O_site>
    """
    number_of_sites = len(mps)

    if not 0 <= site < number_of_sites:
        raise IndexError("site is outside the MPS.")

    operators = [
        identity_operator()
        for _ in range(number_of_sites)
    ]

    operators[site] = np.asarray(operator)

    return expectation_value(mps, operators)


def spin_correlation(mps, first_site, second_site):
    """
    Calculate the z-spin correlation

        <S_z(first_site) S_z(second_site)>.
    """
    number_of_sites = len(mps)

    if first_site == second_site:
        raise ValueError("The two sites must be different.")

    if not 0 <= first_site < number_of_sites:
        raise IndexError("first_site is outside the MPS.")

    if not 0 <= second_site < number_of_sites:
        raise IndexError("second_site is outside the MPS.")

    operators = [
        identity_operator()
        for _ in range(number_of_sites)
    ]

    operators[first_site] = spin_z()
    operators[second_site] = spin_z()

    return expectation_value(mps, operators)