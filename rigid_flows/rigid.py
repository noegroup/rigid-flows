import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass

from flox import geom


@pytree_dataclass(frozen=True)
class InternalCoordinates:
    # see https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/data/tip4pew.xml
    d_OH1: jnp.ndarray = jnp.array(0.09572)
    d_OH2: jnp.ndarray = jnp.array(0.09572)
    a_HOH: jnp.ndarray = jnp.array(1.8242182)

    def asarray(self):
        return jnp.array(
            [
                jnp.array([0.0, 0.0, 0.0]),
                jnp.array(
                    [jnp.sin(-self.a_HOH / 2), 0.0, jnp.cos(-self.a_HOH / 2)]
                )
                * self.d_OH1,
                jnp.array(
                    [jnp.sin(+self.a_HOH / 2), 0.0, jnp.cos(+self.a_HOH / 2)]
                )
                * self.d_OH2,
            ]
        )

    def frame(self):
        u, v = self.asarray()[1:3]
        u = geom.unit(u)
        v = geom.unit(v)
        w = jnp.cross(u, v)
        return jnp.stack([u, v, w])

    def inverse_frame(self):
        return jnp.linalg.inv(self.frame())


def quat_to_mat(q):
    """projects quaternion onto the corresponding
    3x3 rotation matrix

    this is a surjective but not injective map!
    """
    qw, qi, qj, qk = q
    return jnp.array(
        [
            [
                1 - 2 * (qj**2 + qk**2),
                2 * (qi * qj - qk * qw),
                2 * (qi * qk + qj * qw),
            ],
            [
                2 * (qi * qj + qk * qw),
                1 - 2 * (qi**2 + qk**2),
                2 * (qj * qk - qi * qw),
            ],
            [
                2 * (qi * qk - qj * qw),
                2 * (qj * qk + qi * qw),
                1 - 2 * (qi**2 + qj**2),
            ],
        ]
    )


def mat_to_quat(R):
    traces = jnp.array(
        [
            1 + R[0, 0] + R[1, 1] + R[2, 2],
            1 + R[0, 0] - R[1, 1] - R[2, 2],
            1 - R[0, 0] + R[1, 1] - R[2, 2],
            1 - R[0, 0] - R[1, 1] + R[2, 2],
        ]
    )
    weight = jax.nn.softmax(-1.0 / (traces + 1e-6))
    traces = jnp.sqrt(traces + 1e-6) * 2
    out = jnp.array(
        [
            [
                0.25 * traces[0],
                (R[2, 1] - R[1, 2]) / traces[0],
                (R[0, 2] - R[2, 0]) / traces[0],
                (R[1, 0] - R[0, 1]) / traces[0],
            ],
            [
                (R[2, 1] - R[1, 2]) / traces[1],
                0.25 * traces[1],
                (R[0, 1] + R[1, 0]) / traces[1],
                (R[0, 2] + R[2, 0]) / traces[1],
            ],
            [
                (R[0, 2] - R[2, 0]) / traces[2],
                (R[0, 1] + R[1, 0]) / traces[2],
                0.25 * traces[2],
                (R[1, 2] + R[2, 1]) / traces[2],
            ],
            [
                (R[1, 0] - R[0, 1]) / traces[3],
                (R[0, 2] + R[2, 0]) / traces[3],
                (R[1, 2] + R[2, 1]) / traces[3],
                0.25 * traces[3],
            ],
        ]
    )
    out = jnp.sign(out)[:, (0,)] * out
    out = (out * weight[:, None]).sum(0)
    return out


DISTANCE_OM = 0.0125
VIRTUAL_SITE = jnp.array([[0.0, 0.0, DISTANCE_OM]])


@pytree_dataclass(frozen=True)
class Rigid:

    rot: jnp.ndarray  # = jnp.eye(4)[0]
    pos: jnp.ndarray  # = jnp.zeros(3)
    ics: InternalCoordinates  # = InternalCoordinates()
    # other: jnp.ndarray  # = VIRTUAL_SITE

    @staticmethod
    def from_array(inp: jnp.ndarray):
        pos = inp[0]

        arr = inp - pos

        d_OH1 = geom.norm(arr[1])
        d_OH2 = geom.norm(arr[2])
        a_HOH = jnp.arccos(geom.inner(arr[1] / d_OH1, arr[2] / d_OH2))

        ics = InternalCoordinates(
            d_OH1,
            d_OH2,
            a_HOH,
        )

        u = geom.unit(arr[1] - arr[0])
        v = geom.unit(arr[2] - arr[0])
        w = jnp.cross(u, v)
        frame = jnp.stack([u, v, w])
        R = ics.inverse_frame() @ frame

        rot = mat_to_quat(R)

        # other = arr[3:] @ R.T

        return Rigid(rot, pos, ics)

    def asarray(self):
        arr = self.ics.asarray()
        arr = jnp.concatenate([arr, jnp.zeros((1, 3))])
        R = quat_to_mat(self.rot)
        arr = arr @ R
        arr = arr + self.pos
        return arr
