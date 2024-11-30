import pickle

import matplotlib.pyplot as plt
import numpy as np
from laddu.utils.vectors import Vector3, Vector4
from scipy.special import factorial


def breakup_momentum(a: float, b: float, c: float) -> float:
    x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c)
    return np.sqrt(x) / (2 * a)


class Decay:
    def __init__(self, decaying_p4: Vector4, masses: list[float]):
        self.n = len(masses)
        self.budget = decaying_p4.mag - sum(masses)
        self.masses = masses
        self.max_weight = (
            np.power(self.budget, self.n - 2)
            * (np.pi * np.power(2 * np.pi, self.n - 2) / factorial(self.n - 2))
            / decaying_p4.m
        )
        self.decaying_p4 = decaying_p4

    def generate(self, rng: np.random.Generator):
        # set up initial decay vectors and weight
        dec_momenta: list[Vector4] = [Vector4(0.0, 0.0, 0.0, 0.0)] * self.n
        weight = self.max_weight

        # generate sorted random capped by 0 and 1
        r = [0.0, 1.0]
        if self.n > 2:
            r = rng.random(size=(self.n - 2))
            r = sorted(r)
            r = [0.0, *r, 1.0]

        # generate intermediate masses
        int_masses = [0.0] * self.n
        child_masses = 0.0
        for i in range(0, self.n):
            child_masses += self.masses[i]
            int_masses[i] = r[i] * self.budget + child_masses

        # generate momenta magnitudes
        q = [0.0] * self.n
        for i in range(0, self.n - 1):
            q[i] = breakup_momentum(int_masses[i + 1], int_masses[i], self.masses[i + 1])
            weight *= q[i]

        # assign momentum to particle 0
        dec_momenta[0] = Vector4(np.sqrt(q[0] ** 2 + self.masses[0] ** 2), 0.0, 0.0, q[0])

        # assign momentum to other particles
        i = 1
        while True:
            dec_momenta[i] = Vector4(np.sqrt(q[i - 1] ** 2 + self.masses[i] ** 2), 0.0, 0.0, -q[i - 1])

            # rotate by random spherical angles
            cos_y = rng.uniform(-1.0, 1.0)
            sin_y = np.sqrt(1.0 - cos_y**2)
            phi = 2.0 * np.pi * rng.random()
            cos_z = np.cos(phi)
            sin_z = np.sin(phi)
            for j in range(0, i + 1):
                v = dec_momenta[j]
                v_roty = Vector4(
                    v.e,
                    sin_y * v.pz + cos_y * v.px,
                    v.py,
                    cos_y * v.pz - sin_y * v.px,
                )
                v_rotz = Vector4(
                    v_roty.e,
                    cos_z * v_roty.px - sin_z * v_roty.py,
                    sin_z * v_roty.px + cos_z * v_roty.py,
                    v_roty.pz,
                )
                dec_momenta[j] = v_rotz
            if i == self.n - 1:
                break

            # boost to rest frame of parent particle
            beta = q[i] / np.sqrt(q[i] ** 2 + int_masses[i] ** 2)
            for j in range(0, i + 1):
                v_boosted = dec_momenta[j].boost(Vector3(0.0, 0.0, -beta))
                dec_momenta[j] = v_boosted
            i += 1

        # boost to original frame of decaying particle
        for j in range(0, self.n):
            v_boosted = dec_momenta[j].boost_along(self.decaying_p4)
            dec_momenta[j] = v_boosted
        return dec_momenta, weight


def main():
    target = Vector4(0.938, 0.0, 0.0, 0.0)
    beam = Vector4(0.65, 0.0, 0.0, 0.65)
    w = beam + target
    masses = [
        0.938,
        0.139,
        0.139,
    ]
    dec = Decay(w, masses)
    m_p_pim = []
    m_p_pip = []
    weights = []
    rng = np.random.default_rng()
    for _ in range(100000):
        vecs, weight = dec.generate(rng)
        m_p_pim.append((vecs[0] + vecs[1]).m2)
        m_p_pip.append((vecs[0] + vecs[2]).m2)
        weights.append(weight)

    with open("rust_values.pkl", "rb") as f:
        rust_data = pickle.load(f)

    rust_m_p_pip = rust_data["m_p_pip"]
    rust_m_p_pim = rust_data["m_p_pim"]
    rust_weights = rust_data["weights"]

    _, ax = plt.subplots(ncols=2, sharey=True)
    ax[0].hist2d(m_p_pim, m_p_pip, bins=[50, 50], range=[[1.1, 1.8], [1.1, 1.8]], weights=weights)
    ax[0].set_title("Python")
    ax[0].set_xlabel(r"$M(p\pi^-)^2$")
    ax[0].set_ylabel(r"$M(p\pi^+)^2$")
    ax[1].hist2d(rust_m_p_pim, rust_m_p_pip, bins=[50, 50], range=[[1.1, 1.8], [1.1, 1.8]], weights=rust_weights)
    ax[1].set_title("Rust")
    ax[1].set_xlabel(r"$M(p\pi^-)^2$")
    ax[1].set_ylabel(r"$M(p\pi^+)^2$")

    plt.savefig("output.png")


if __name__ == "__main__":
    main()
