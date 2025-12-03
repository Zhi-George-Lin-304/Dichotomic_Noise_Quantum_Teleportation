import numpy as np

from .operators import I2, X, Z, op_on_qubit

P0 = np.array([[1, 0],
               [0, 0]], dtype=complex)
P1 = np.array([[0, 0],
               [0, 1]], dtype=complex)


def projector_on_01(a: int, b: int) -> np.ndarray:
    """
    Projector onto |a,b⟩ on qubits 0 and 1 (indices 0 and 1).
    """
    Pa = P0 if a == 0 else P1
    Pb = P0 if b == 0 else P1
    ops = [Pa, Pb, I2]  # qubit 2 is untouched
    M = ops[0]
    for q in range(1, 3):
        M = np.kron(M, ops[q])
    return M


def correction_on_2(a: int, b: int) -> np.ndarray:
    """
    Teleportation feedforward: apply X^b Z^a on qubit 2 (index 2).
    """
    U = np.eye(2, dtype=complex)
    if b == 1:
        U = X @ U
    if a == 1:
        U = Z @ U
    return op_on_qubit(U, 2, n=3)


def meas_and_feedforward(rho: np.ndarray) -> np.ndarray:
    """
    Apply projective measurement on qubits 0,1 + classical
    feedforward on qubit 2, averaged over measurement outcomes.
    """
    rho_out = np.zeros_like(rho, dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            M = projector_on_01(a, b)
            C = correction_on_2(a, b)
            rho_ab = M @ rho @ M              # unnormalized post-measurement
            rho_out += C @ rho_ab @ C.conj().T
    return rho_out


def trace_out_qubits_0_1(rho: np.ndarray) -> np.ndarray:
    """
    Partial trace over qubits 0 and 1, returning 2x2 density matrix of qubit 2.
    Basis ordering: |q0 q1 q2⟩ with q0 = qubit 0, q1 = qubit 1, q2 = qubit 2.
    """
    rho2 = np.zeros((2, 2), dtype=complex)
    for a in (0, 1):      # qubit 0
        for b in (0, 1):  # qubit 1
            for c in (0, 1):
                for c2 in (0, 1):
                    i = (a << 2) + (b << 1) + c
                    j = (a << 2) + (b << 1) + c2
                    rho2[c, c2] += rho[i, j]
    return rho2