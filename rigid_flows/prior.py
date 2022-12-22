import math
from turtle import forward

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp  # type: ignore
from jax import Array

from experiments.rigid_flows.rigid_flows.system import SimulationBox
from flox import geom
from flox._src.flow import Transformed, rigid

from .data import Data

KeyArray = jax.random.PRNGKeyArray | jnp.ndarray


# Computes I_{nu+1}(x) / I_{nu}(x).
# Amos (1974), Computation of modified Bessel functions and
# their ratios, Mathematics of Computation, 28 (125), 235-251
def bessel_ratio(nu, x, threshold=1e-6):
    l = []
    res = np.array(2.0, dtype=np.float64)
    k = 0
    while True:
        s = nu + k + 0.5
        l.append(x / (s + np.hypot(s + 1, x)))
        m = k
        while m > 0:
            r = np.sqrt(l[m] / l[m - 1])
            s = nu + m
            l[m - 1] = x / (s + np.hypot(s, x * r))
            m -= 1

        err = np.abs(l[0] - res)
        if err < threshold:
            return res
        res = l[0]
        k += 1
        if k > 100:
            raise ValueError(f"iteration did not converge {err}")


def mle_kappa(r, p, threshold=1e-6, maxiters=10000):
    k = r * (p - r * r) / (1 - r * r)
    ak = 1
    delta = 1
    s = 0
    while np.abs(delta) > threshold:
        ak = bessel_ratio(p / 2 - 1, k)
        delta = (ak - r) / (1 - ak * ak - (p - 1) / k * ak)
        k -= delta
        s += 1
        if s > maxiters:
            raise ValueError(f"iteration did not converge {delta}")
    return k


def mle_loc(q):
    x = np.mean(q, axis=0)
    r = np.sqrt(np.square(x).sum())
    return x / r, r


def mle_vmf_params(q):
    q = np.array(q, dtype=np.float64)
    loc, r = mle_loc(q)
    scale = mle_kappa(r, q.shape[-1])
    return loc, scale


def smooth_maximum(a, bins=1000, sigma=10000, window=1000):
    freqs, bins = jnp.histogram(a, bins=bins)
    gx = np.arange(-4 * sigma, 4 * sigma, window)
    gaussian = np.exp(-((gx / sigma) ** 2) / 2)
    freqs = jnp.convolve(freqs, gaussian, mode="same")
    return bins[jnp.argmax(freqs)]


# from tensorflow_probability.substrates import jax as tfp


class RotationPrior(eqx.Module):
    loc: Array
    scale: Array
    vmf: tfp.distributions.VonMisesFisher

    def __init__(self, data: Data):
        def quat(x: Array):
            frames = x.reshape(-1, 4, 3)[:, :3, :]
            q, *_ = jax.vmap(rigid.from_euclidean)(frames)
            return q

        quats = jax.vmap(quat)(data.pos)
        sign = jnp.sign(jnp.sum(quats * quats[(0,), :, :], axis=-1))
        quats = quats * sign[:, :, None]

        locs = []
        scales = []
        for i in range(quats.shape[1]):
            qs = quats[:, i]
            loc, scale = mle_vmf_params(qs)
            locs.append(loc)
            scales.append(scale)

        self.loc = jnp.stack(locs)
        self.scale = jnp.stack(scales)
        self.vmf = tfp.distributions.VonMisesFisher(self.loc, self.scale)

    def sample(self, seed: KeyArray):
        # return jax.vmap(geom.unit)(jax.random.normal(key, shape=self.loc.shape))
        return self.vmf.sample(seed=seed)

    def log_prob(self, x: Array):
        parts = jnp.stack(
            [
                self.vmf.log_prob(x) + jnp.log(0.5),
                self.vmf.log_prob(-x) + jnp.log(0.5),
            ]
        )
        return jax.nn.logsumexp(parts)


class PositionPrior(eqx.Module):

    box: SimulationBox

    mean: Array
    diag: Array
    cov_sqrt: Array
    inv_cov_sqrt: Array

    com_mean: Array
    com_std: Array

    def __init__(self, data: Data):
        self.box = SimulationBox(data.box)
        oxy = data.pos.reshape(data.pos.shape[0], -1, 4, 3)[:, :, 0]

        self.mean = jax.vmap(
            jax.vmap(smooth_maximum, in_axes=1, out_axes=0),
            in_axes=2,
            out_axes=1,
        )(oxy)

        # r = oxy.reshape(oxy.shape[0], -1)

        r = jax.vmap(
            lambda x: geom.Torus(self.box.size).tangent(x, x - self.mean)
        )(oxy)
        r = r.reshape(r.shape[0], -1)
        # self.mean = jnp.mean(r, axis=0).reshape(oxy.shape[1:])

        # unfold torus
        oxy = self.mean[None] + geom.Torus(self.box.size).tangent(
            oxy, oxy - self.mean[None]
        )
        self.com_mean = jnp.mean(oxy, axis=(0, 1))
        self.com_std = jnp.std(oxy, axis=(0, 1))

        C = jnp.cov(r.T)
        D, U = jnp.linalg.eigh(C)
        D = jnp.sqrt(D)
        self.diag = D
        self.cov_sqrt = jnp.diag(D) @ U.T
        self.inv_cov_sqrt = U @ jnp.diag(1.0 / (D + 1e-6))

    def forward(self, x: Array):
        x = x.reshape(self.mean.shape)
        x = x - self.mean
        x = x.reshape(-1)
        x = x @ self.inv_cov_sqrt
        x = x.reshape(self.mean.shape)
        ldj = jnp.sum(-jnp.log(self.diag)) * math.prod(self.mean.shape[:-1])
        return Transformed(x, ldj)

    def inverse(self, x: Array):
        x = x.reshape(-1)
        x = self.cov_sqrt.T @ x
        x = x.reshape(self.mean.shape)
        x = x + self.mean
        ldj = jnp.sum(jnp.log(self.diag)) * math.prod(self.mean.shape[:-1])
        return Transformed(x, ldj)

    def sample(self, *, seed: KeyArray):
        r = jax.random.normal(seed, shape=(math.prod(self.mean.shape),))
        r = (self.cov_sqrt.T @ r).reshape(self.mean.shape)
        r = r + self.mean
        return r

    def log_prob(self, x: Array):
        x = x.reshape(self.mean.shape)
        diff = (x - self.mean).reshape(-1)
        return -0.5 * jnp.square(diff @ self.inv_cov_sqrt).sum()
