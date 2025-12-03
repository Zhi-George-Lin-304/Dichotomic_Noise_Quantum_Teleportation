import numpy as np

# Basic 1-qubit operators
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)
Z = np.array([[1,  0],
              [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1,  1],
                               [1, -1]], dtype=complex)


def op_on_qubit(U: np.ndarray, target: int, n: int = 3) -> np.ndarray:
    """
    Embed a 2x2 matrix U as acting on qubit 'target' (0..n-1)
    in an n-qubit Hilbert space, with identity on others.
    Qubit 0 is the leftmost factor in the Kronecker product.
    """
    ops = []
    for q in range(n):
        ops.append(U if q == target else I2)
    out = ops[0]
    for q in range(1, n):
        out = np.kron(out, ops[q])
    return out


def cnot(control: int, target: int, n: int = 3) -> np.ndarray:
    """
    Controlled-NOT on n qubits with given control and target (0-based indices).
    """
    P0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)
    P1 = np.array([[0, 0],
                   [0, 1]], dtype=complex)

    ops0, ops1 = [], []
    for q in range(n):
        if q == control:
            ops0.append(P0)
            ops1.append(P1)
        elif q == target:
            ops0.append(I2)
            ops1.append(X)
        else:
            ops0.append(I2)
            ops1.append(I2)

    U0 = ops0[0]
    U1 = ops1[0]
    for q in range(1, n):
        U0 = np.kron(U0, ops0[q])
        U1 = np.kron(U1, ops1[q])
    return U0 + U1


def liouvillian_from_H(H: np.ndarray) -> np.ndarray:
    """
    Build Liouvillian L such that d/dt vec(rho) = L vec(rho)
    using column-stacking vec:
        L = -i (I ⊗ H - H^T ⊗ I).
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    return -1j * (np.kron(I, H) - np.kron(H.conj().T, I))
