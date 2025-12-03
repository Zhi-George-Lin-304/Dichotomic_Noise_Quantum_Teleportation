import numpy as np

from .teleportation import teleport_with_noise


def teleportation_fidelity_for_state(
    psi: np.ndarray,
    Delta1: float,
    Delta2: float,
    Gamma1: float,
    Gamma2: float,
    T: float,
    J: float = 0.0,
) -> float:
    """
    Fidelity F = ⟨psi| rho_out |psi⟩ for a given input |psi⟩,
    where rho_out is the final state of qubit 2.
    """
    rho_out = teleport_with_noise(psi, Delta1, Delta2, Gamma1, Gamma2, T, J=J)
    psi = psi / np.linalg.norm(psi)
    return float(np.real(np.vdot(psi, rho_out @ psi)))


def average_teleportation_fidelity(
    Delta1: float,
    Delta2: float,
    Gamma1: float,
    Gamma2: float,
    T: float,
    J: float = 0.0,
) -> float:
    """
    Approximate the average teleportation fidelity by averaging over
    6 cardinal states on the Bloch sphere: |0>,|1>,|+>,|->,|+i>,|-i>.
    """
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    ketp = (ket0 + ket1) / np.sqrt(2)
    ketm = (ket0 - ket1) / np.sqrt(2)
    ketpi = (ket0 + 1j * ket1) / np.sqrt(2)
    ketmi = (ket0 - 1j * ket1) / np.sqrt(2)

    states = [ket0, ket1, ketp, ketm, ketpi, ketmi]

    F_sum = 0.0
    for psi in states:
        F_sum += teleportation_fidelity_for_state(
            psi, Delta1, Delta2, Gamma1, Gamma2, T, J=J
        )
    return F_sum / len(states)
