import numpy as np
import matplotlib.pyplot as plt

from teleport_rtn import average_teleportation_fidelity


def main():
    # Base parameters (arb. units)
    Delta2 = 1.0
    Delta3 = 1.0
    Gamma2 = 0.5
    Gamma3 = 0.5
    J_exch = 1.0

    # (a) F_avg vs storage time T
    Ts = np.linspace(0.0, 5.0, 40)
    F_T = []
    for T in Ts:
        F = average_teleportation_fidelity(
            Delta2, Delta3, Gamma2, Gamma3, T, J=J_exch
        )
        F_T.append(F)
        print(f"T={T:.2f}, F_avg={F:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(Ts, F_T, "o-", lw=2, label="F_avg(T)")
    plt.axhline(2/3, color="k", ls="--", label="classical limit 2/3")
    plt.xlabel("Storage time T (arb. units)")
    plt.ylabel("Average teleportation fidelity")
    plt.ylim(0.4, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (b) F_avg vs Γ (Γ2=Γ3=Γ)
    Gammas = np.linspace(0.05, 2.0, 20)
    T_fixed = 2.0
    F_G = []
    for G in Gammas:
        F_G.append(
            average_teleportation_fidelity(
                Delta2, Delta3, G, G, T_fixed, J=J_exch
            )
        )

    plt.figure(figsize=(6, 4))
    plt.plot(Gammas, F_G, "o-", lw=2)
    plt.axhline(2/3, color="k", ls="--")
    plt.xlabel(r"Flipping rate $\Gamma$ (arb. units)")
    plt.ylabel(r"$F$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (c) F_avg vs Δ (Δ2=Δ3=Δ)
    Deltas = np.linspace(0.0, 2.0, 20)
    F_D = []
    for D in Deltas:
        F_D.append(
            average_teleportation_fidelity(
                D, D, Gamma2, Gamma3, T_fixed, J=J_exch
            )
        )

    plt.figure(figsize=(6, 4))
    plt.plot(Deltas, F_D, "o-", lw=2)
    plt.axhline(2/3, color="k", ls="--")
    plt.xlabel(r"Noise amplitude $\Delta$ (arb. units)")
    plt.ylabel(r"$F$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
