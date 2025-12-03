import numpy as np

from .operators import X, Y, Z, op_on_qubit

# 3-qubit global Z operators (qubits 0,1,2)
Z0 = op_on_qubit(Z, 0, n=3)
Z1 = op_on_qubit(Z, 1, n=3)
Z2 = op_on_qubit(Z, 2, n=3)

# X, Y on qubits 1 and 2 (the Bell pair)
X1_op = op_on_qubit(X, 1, n=3)
X2_op = op_on_qubit(X, 2, n=3)
Y1_op = op_on_qubit(Y, 1, n=3)
Y2_op = op_on_qubit(Y, 2, n=3)

I_sys = np.eye(8, dtype=complex)


def H0_exchange(J: float) -> np.ndarray:
    """
    Effective XY exchange Hamiltonian between qubits 1 and 2:

        H0 = J/2 (X1 X2 + Y1 Y2)

    acting on the full 3-qubit Hilbert space (qubits 0,1,2).
    """
    if J == 0.0:
        return np.zeros((8, 8), dtype=complex)
    return 0.5 * J * (X1_op @ X2_op + Y1_op @ Y2_op)
