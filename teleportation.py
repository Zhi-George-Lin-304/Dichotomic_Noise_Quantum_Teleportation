import numpy as np

from .operators import op_on_qubit, cnot, H
from .noise import propagate_noise
from .measurement import meas_and_feedforward, trace_out_qubits_0_1


def teleport_with_noise(
    psi: np.ndarray,
    Delta1: float,
    Delta2: float,
    Gamma1: float,
    Gamma2: float,
    T: float,
    J: float = 0.0,
) -> np.ndarray:
    """
    Full three-qubit teleportation circuit (0-based qubit labels):

    - Qubit 0: input state |psi⟩
    - Qubits 1,2: initially in |00⟩ and then entangled into a Bell pair
    - Bell pair preparation on qubits 1,2 (H on 2, then CNOT 2→1)
    - Storage under RTN + exchange H0 on qubits 1,2 for time T
    - Bell measurement on qubits 0,1 (CNOT 0→1, H on 0)
    - Projective measurement + classical feedforward on qubit 2

    Returns the final 2x2 density matrix of qubit 2.
    """
    zero = np.array([1, 0], dtype=complex)

    # Initial state: |psi⟩_0 ⊗ |0⟩_1 ⊗ |0⟩_2
    state0 = np.kron(np.kron(psi, zero), zero)    # 8-vector
    rho0 = np.outer(state0, state0.conj())        # 8x8

    # Bell pair preparation on qubits 1,2: H on 2, then CNOT 2→1
    U_Bell = cnot(control=2, target=1, n=3) @ op_on_qubit(H, 2, n=3)
    rho1 = U_Bell @ rho0 @ U_Bell.conj().T

    # Noise + exchange during storage on qubits 1 and 2
    rho2 = propagate_noise(rho1, Delta1, Delta2, Gamma1, Gamma2, T, J=J)

    # Bell measurement on qubits 0,1: CNOT 0→1 then H on 0
    U_meas = op_on_qubit(H, 0, n=3) @ cnot(control=0, target=1, n=3)
    rho3_full = U_meas @ rho2 @ U_meas.conj().T

    # Measurement + feedforward on qubit 2
    rho4_full = meas_and_feedforward(rho3_full)

    # Reduced state of qubit 2 (trace out 0 and 1)
    rho_out = trace_out_qubits_0_1(rho4_full)
    return rho_out