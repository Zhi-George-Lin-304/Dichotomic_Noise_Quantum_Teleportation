import numpy as np
from scipy.linalg import block_diag, expm

from .operators import liouvillian_from_H
from .hamiltonian import Z1, Z2, H0_exchange


def build_noise_liouvillian(
    Delta1: float,
    Delta2: float,
    Gamma1: float,
    Gamma2: float,
    J: float = 0.0,
) -> np.ndarray:
    """
    Build the extended Liouvillian for RTN dephasing on qubits 1 and 2
    during a storage time T, optionally including an exchange H0 between
    these two qubits.

    Noise variables: k1, k2 ∈ {+1, -1}.
    Configurations: (+1,+1), (+1,-1), (-1,+1), (-1,-1).
    """
    configs = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    L_blocks = []
    H0 = H0_exchange(J)

    # System Liouvillians (block-diagonal over noise configs)
    for (k1, k2) in configs:
        Hk = H0 + (Delta1 * k1 * Z1 + Delta2 * k2 * Z2)
        Lk = liouvillian_from_H(Hk)      # 64 x 64
        L_blocks.append(Lk)

    L_sys = block_diag(*L_blocks)        # 256 x 256

    # Classical generator for two independent RTN processes
    Q1 = np.array([[-Gamma1,  Gamma1],
                   [ Gamma1, -Gamma1]], dtype=float)
    Q2 = np.array([[-Gamma2,  Gamma2],
                   [ Gamma2, -Gamma2]], dtype=float)

    I2c = np.eye(2)
    Q_tot = np.kron(Q1, I2c) + np.kron(I2c, Q2)   # 4 x 4

    # Lift classical generator to extended Liouville space
    I_L = np.eye(64, dtype=complex)
    L_classical = np.kron(Q_tot, I_L)            # 256 x 256

    # Total extended Liouvillian
    return L_sys + L_classical


def propagate_noise(
    rho0: np.ndarray,
    Delta1: float,
    Delta2: float,
    Gamma1: float,
    Gamma2: float,
    T: float,
    J: float = 0.0,
) -> np.ndarray:
    """
    Evolve a 3-qubit density matrix rho0 under RTN on qubits 1,2
    plus an optional exchange H0 between qubits 1 and 2 for time T,
    using the extended Liouvillian.
    """
    L_ext = build_noise_liouvillian(Delta1, Delta2, Gamma1, Gamma2, J=J)

    # Initial extended state: stationary distribution over noise configs
    vec_rho0 = rho0.reshape(64, order="F")
    blocks = [0.25 * vec_rho0 for _ in range(4)]
    R0 = np.concatenate(blocks)          # 256-vector

    U_ext = expm(L_ext * T)
    R_T = U_ext @ R0                     # 256-vector

    # Sum over noise configurations to get noise-averaged rho(T)
    R_T_blocks = R_T.reshape(4, 64)
    vec_rho_T = np.sum(R_T_blocks, axis=0)
    rho_T = vec_rho_T.reshape(8, 8, order="F")
    return rho_T
