# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Plane cross-section model of a three-core trefoil bundle with contact.

The three cores, an optional interstitial filler and the first surrounding
ring (bedding or inner sheath) are discretised with isoparametric six-node
triangles in generalised plane strain. Under axisymmetric loads (uniform
pressure on the ring, common extension and twist, uniform layer
temperatures) the trefoil has three mirror planes, so a 60 degree sector
containing half of one core is solved and every energy is multiplied by six.

Contact is frictionless and unilateral. Core-to-core contact lies on the
60 degree mirror plane and becomes contact of the core with that rigid,
frictionless plane. Core-to-ring contact is node-to-segment, with the ring
bore nodes as slaves against the core outer boundary. Filler-to-core
and filler-to-ring contact is node-to-node on coincident reference nodes.
All gaps are linearised about the reference geometry (small displacement).

Each core is a helix of lay angle beta at pitch radius r_c. Its uniform
longitudinal strain is cos²beta e + r_c cos beta sin beta tau +
sin²beta u_c / r_c, where u_c is the radial displacement of the core centre
(for a hollow core, the mean of the two bore points on the mirror axis).
The cross-section plane of the core is taken as the cable cross-section
(an error of order 1 - cos beta). Layers inside a core are bonded. Ring and
filler share the cable extension. Twist rigidities of the ring and filler
assume no interface slip; each core adds its own St Venant rigidity with
twist cos²beta tau. No ovalisation of the ring from non-uniform outer
pressure is modelled: layers outside the ring are coupled through the mean
radial displacement of its outer surface and apply uniform pressure.

A small grounded stabilising stiffness is added to every in-plane degree of
freedom so that a core or filler that is separated from all its neighbours
does not make the stiffness singular. Its default value is 1e-10 times the
smallest modulus [Pa] per node; results are insensitive to it once a load
path is closed (checked by the tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError
from cablexsec.mechanics.concentric_section import ElasticAnnulus

__all__ = [
    "ElasticFiller",
    "TrefoilBundle",
    "TrefoilModel",
    "build_trefoil_model",
]

FloatArray = NDArray[np.float64]
SECTOR = math.pi / 3
SYMMETRY_FACTOR = 6.0

# Six-point degree-four rule on the reference triangle (Dunavant).
_A, _B = 0.445948490915965, 0.091576213509771
_WA, _WB = 0.223381589678011 / 2, 0.109951743655322 / 2
_QP = np.array(
    [
        [_A, _A],
        [1 - 2 * _A, _A],
        [_A, 1 - 2 * _A],
        [_B, _B],
        [1 - 2 * _B, _B],
        [_B, 1 - 2 * _B],
    ]
)
_QW = np.array([_WA, _WA, _WA, _WB, _WB, _WB])


@dataclass(frozen=True)
class ElasticFiller:
    """Isotropic interstitial filler occupying the outer interstices."""

    name: str
    youngs_modulus_Pa: float
    poisson_ratio: float
    free_thermal_strain: float = 0.0


@dataclass(frozen=True)
class TrefoilBundle:
    """Three identical cores in trefoil inside a surrounding ring.

    core_layers are annuli about the core centre, inner to outer, contiguous
    and bonded. ring_layers are contiguous annuli about the cable axis, bonded
    to each other, whose bore faces the cores (bedding, binder, inner
    sheath). Free strains of the layer objects include temperature and any
    flooded-pressure strain. The filler, when given, fills the three outer
    interstices between cores and ring; the central interstice is void.
    """

    core_layers: tuple[ElasticAnnulus, ...]
    core_centre_radius_m: float
    ring_layers: tuple[ElasticAnnulus, ...]
    core_lay_angle_rad: float = 0.0
    filler: ElasticFiller | None = None
    angular_cells: int = 48
    radial_cell_size_m: float | None = None
    overlap_tolerance: float = 5.0e-3
    stabilisation: float = 1.0e-10

    @property
    def core_radius_m(self) -> float:
        return self.core_layers[-1].outer_radius_m

    @property
    def envelope_radius_m(self) -> float:
        return self.core_centre_radius_m + self.core_radius_m


@dataclass(frozen=True)
class ContactFamily:
    """Rows of one contact type, with the physical force scale per row."""

    name: str
    rows: slice
    # Physical force = factor * sum of row reactions: per core for core-ring
    # and filler-core, per core pair for core-core, whole ring for
    # filler-ring. Sums of normal-force magnitudes, not vector resultants.
    force_factor: float


@dataclass(frozen=True)
class TrefoilModel:
    """Assembled full-section stiffness of the bundle (sector times six).

    Degrees of freedom are [extension, twist, in-plane nodal displacements].
    Contact rows C give gaps g = g0 + C u (full numbering); a positive
    reaction pushes the surfaces apart. outer_weights gives the mean radial
    displacement of the ring outer surface. constrained maps reduced to full
    in-plane freedoms for the mirror conditions.
    """

    stiffness: sp.csr_matrix
    free_strain_load: FloatArray
    contact: sp.csr_matrix
    gap: FloatArray
    families: tuple[ContactFamily, ...]
    outer_weights: sp.csr_matrix  # 1 x n
    transform: sp.csr_matrix  # n x n_reduced
    nodes: FloatArray
    core_centre_dofs: tuple[int, int]
    core_axial_rigidity_N: float
    core_axial_free_strain: float
    core_lay_angle_rad: float
    core_centre_radius_m: float
    ring_outer_radius_m: float
    adjusted_overlaps_m: dict[str, float]
    filled_core_gaps: tuple[str, ...]
    element_groups: dict[str, NDArray[np.int64]]
    element_nodes: NDArray[np.int64]
    element_moduli: FloatArray
    element_poisson: FloatArray
    element_free_strain: FloatArray
    element_axial_map: FloatArray  # (ne, 4): ext, twist, two centre dofs
    element_layers: tuple[str, ...] = ()
    """Name of the layer of each element (a core layer, ring layer or the
    filler); a radial gap filled with a neighbour's material is that layer's."""
    element_transverse_strain: FloatArray | None = None
    """In-plane free strain of each element added to its isotropic one (a
    ring layer's frozen extrusion shrinkage); ``None`` is zero."""

    @property
    def size(self) -> int:
        return int(self.stiffness.shape[0])


def _shape(xi: float, eta: float) -> tuple[FloatArray, FloatArray]:
    l1, l2, l3 = 1 - xi - eta, xi, eta
    n = np.array(
        [
            l1 * (2 * l1 - 1),
            l2 * (2 * l2 - 1),
            l3 * (2 * l3 - 1),
            4 * l1 * l2,
            4 * l2 * l3,
            4 * l3 * l1,
        ]
    )
    # Derivatives with respect to (xi, eta); dl1 = (-1, -1).
    d = np.array(
        [
            [-(4 * l1 - 1), -(4 * l1 - 1)],
            [4 * l2 - 1, 0.0],
            [0.0, 4 * l3 - 1],
            [4 * (l1 - l2), -4 * l2],
            [4 * l3, 4 * l2],
            [-4 * l3, 4 * (l1 - l3)],
        ]
    )
    return n, d


_SHAPES = [_shape(*p) for p in _QP]


class _NodeTable:
    def __init__(self) -> None:
        self.keys: dict[tuple[object, ...], int] = {}
        self.xy: list[tuple[float, float]] = []

    def get(self, key: tuple[object, ...], xy: tuple[float, float]) -> int:
        index = self.keys.get(key)
        if index is None:
            index = len(self.xy)
            self.keys[key] = index
            self.xy.append(xy)
        return index


def _cells(thickness: float, size: float) -> int:
    return max(1, math.ceil(thickness / size - 1e-9))


def _contiguous_core(
    layers: tuple[ElasticAnnulus, ...],
) -> tuple[list[ElasticAnnulus], list[str]]:
    """Close unlisted radial gaps inside a core with the softer neighbour.

    Unlisted core layers are normally semiconducting screens, which are
    polymeric; the softer adjacent material is extended across the gap.
    """
    result: list[ElasticAnnulus] = []
    filled: list[str] = []
    for layer in layers:
        if not result:
            result.append(layer)
            continue
        previous = result[-1].outer_radius_m
        if layer.inner_radius_m < previous - 1e-12:
            raise CableInputError("core layers overlap")
        if layer.inner_radius_m > previous + 1e-12:
            gap = (previous, layer.inner_radius_m)
            if result[-1].youngs_modulus_Pa < layer.youngs_modulus_Pa:
                donor = result[-1]
                result[-1] = replace(donor, outer_radius_m=gap[1])
            else:
                donor = layer
                layer = replace(layer, inner_radius_m=gap[0])
            filled.append(
                f"from {gap[0] * 1e3:.4g} to {gap[1] * 1e3:.4g} mm radius filled "
                f"with the material of the {donor.name.replace('_', ' ')}"
            )
        result.append(layer)
    return result, filled


def _validate(bundle: TrefoilBundle) -> None:
    if not bundle.core_layers:
        raise CableInputError("a trefoil core needs at least one layer")
    if not bundle.ring_layers:
        raise CableInputError("a trefoil bundle needs a surrounding ring")
    for inner, outer in zip(
        bundle.ring_layers[:-1], bundle.ring_layers[1:], strict=True
    ):
        if abs(outer.inner_radius_m - inner.outer_radius_m) > 1e-12:
            raise CableInputError("ring layers must be contiguous")
    for layer in (*bundle.core_layers, *bundle.ring_layers):
        values = [
            layer.inner_radius_m,
            layer.outer_radius_m,
            layer.youngs_modulus_Pa,
            layer.poisson_ratio,
            layer.free_thermal_strain,
        ]
        if not np.isfinite(values).all():
            raise CableInputError("trefoil layer data must be finite")
        if layer.outer_radius_m <= layer.inner_radius_m or layer.inner_radius_m < 0:
            raise CableInputError("trefoil layers need positive thickness")
        if layer.youngs_modulus_Pa <= 0 or not -1 < layer.poisson_ratio < 0.5:
            raise CableInputError("trefoil moduli must be positive, nu in (-1, 0.5)")
    if bundle.filler is not None:
        f = bundle.filler
        if not (
            math.isfinite(f.youngs_modulus_Pa)
            and f.youngs_modulus_Pa > 0
            and -1 < f.poisson_ratio < 0.5
            and math.isfinite(f.free_thermal_strain)
        ):
            raise CableInputError("filler modulus and Poisson ratio are invalid")
    if (
        isinstance(bundle.angular_cells, bool)
        or not isinstance(bundle.angular_cells, int)
        or bundle.angular_cells < 12
        or bundle.angular_cells % 6
    ):
        raise CableInputError("angular_cells must be a multiple of six, >= 12")
    if not abs(bundle.core_lay_angle_rad) < math.pi / 4:
        raise CableInputError("core lay angle must be below 45 degrees")
    if not bundle.ring_layers[0].inner_radius_m > bundle.core_centre_radius_m:
        raise CableInputError("ring bore must enclose the core centres")


def build_trefoil_model(bundle: TrefoilBundle) -> TrefoilModel:
    """Mesh and assemble the sector model; see the module notes."""
    _validate(bundle)
    layers, filled = _contiguous_core(bundle.core_layers)
    a = layers[-1].outer_radius_m
    rc = bundle.core_centre_radius_m
    ring = bundle.ring_layers[0]
    ring_outer = bundle.ring_layers[-1].outer_radius_m
    beta = bundle.core_lay_angle_rad
    cb, sb = math.cos(beta), math.sin(beta)
    n_phi = bundle.angular_cells
    n_theta = 5 * n_phi // 6
    size = bundle.radial_cell_size_m or a / 10
    if not size > 0:
        raise CableInputError("radial cell size must be positive")
    nodes = _NodeTable()
    elements: list[list[int]] = []
    groups: dict[str, list[int]] = {}
    material: list[tuple[float, float, float, float]] = []
    axial: list[tuple[float, ...]] = []
    owners: list[str] = []

    # Quadratic angular grid of the half core (half-index j = 0..2 n_phi).
    phis = np.linspace(0.0, math.pi, 2 * n_phi + 1)
    # Radial half-grid of the core, by layer.
    rho_half: list[float] = [layers[0].inner_radius_m]
    layer_of_cell: list[int] = []
    for k, layer in enumerate(layers):
        cells = _cells(layer.outer_radius_m - layer.inner_radius_m, size)
        edges = np.linspace(layer.inner_radius_m, layer.outer_radius_m, cells + 1)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            rho_half.extend([0.5 * (lo + hi), hi])
            layer_of_cell.append(k)
    core_outer = len(rho_half) - 1

    def core_node(i: int, j: int) -> int:
        rho = rho_half[i]
        if rho == 0.0:
            return nodes.get(("core", 0, 0), (rc, 0.0))
        phi = phis[j]
        return nodes.get(
            ("core", i, j), (rc + rho * math.cos(phi), rho * math.sin(phi))
        )

    def add(group: str, conn: list[int], layer: object, ext: tuple[float, ...]) -> None:
        groups.setdefault(group, []).append(len(elements))
        elements.append(conn)
        material.append(
            (
                layer.youngs_modulus_Pa,  # type: ignore[attr-defined]
                layer.poisson_ratio,  # type: ignore[attr-defined]
                layer.free_thermal_strain,  # type: ignore[attr-defined]
                getattr(layer, "free_transverse_strain", 0.0),
            )
        )
        axial.append(ext)
        owners.append(str(layer.name))  # type: ignore[attr-defined]

    core_axial = (cb * cb, rc * cb * sb, 0.5 * sb * sb / rc, 0.5 * sb * sb / rc)
    for c, k in enumerate(layer_of_cell):
        i0, i1 = 2 * c, 2 * c + 2
        for q in range(n_phi):
            j0, j1 = 2 * q, 2 * q + 2
            if rho_half[i0] == 0.0:
                fan = [
                    core_node(0, 0),
                    core_node(i1, j0),
                    core_node(i1, j1),
                    core_node(i0 + 1, j0),
                    core_node(i1, j0 + 1),
                    core_node(i0 + 1, j1),
                ]
                add("core", fan, layers[k], core_axial)
                continue
            for tri in (
                [
                    (i0, j0),
                    (i1, j0),
                    (i1, j1),
                    (i0 + 1, j0),
                    (i1, j0 + 1),
                    (i0 + 1, j0 + 1),
                ],
                [
                    (i0, j0),
                    (i1, j1),
                    (i0, j1),
                    (i0 + 1, j0 + 1),
                    (i0 + 1, j1),
                    (i0, j0 + 1),
                ],
            ):
                add("core", [core_node(*p) for p in tri], layers[k], core_axial)

    # Ring sector about the cable axis; its bore nodes match the filler map.
    thetas = 0.4 * phis[: 2 * n_theta + 1]  # theta = (pi/3) / (5 pi/6) * phi
    ring_r_list: list[float] = [ring.inner_radius_m]
    ring_layer_of_cell: list[ElasticAnnulus] = []
    for layer in bundle.ring_layers:
        cells = _cells(layer.outer_radius_m - layer.inner_radius_m, size)
        edges = np.linspace(layer.inner_radius_m, layer.outer_radius_m, cells + 1)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            ring_r_list.extend([0.5 * (lo + hi), hi])
            ring_layer_of_cell.append(layer)
    ring_r = np.array(ring_r_list)
    ring_cells = len(ring_layer_of_cell)

    def ring_node(i: int, j: int) -> int:
        r, t = ring_r[i], thetas[j]
        return nodes.get(("ring", i, j), (r * math.cos(t), r * math.sin(t)))

    ring_axial = (1.0, 0.0, 0.0, 0.0)
    for c in range(ring_cells):
        i0, i1 = 2 * c, 2 * c + 2
        for q in range(n_theta):
            j0, j1 = 2 * q, 2 * q + 2
            for tri in (
                [
                    (i0, j0),
                    (i1, j0),
                    (i1, j1),
                    (i0 + 1, j0),
                    (i1, j0 + 1),
                    (i0 + 1, j0 + 1),
                ],
                [
                    (i0, j0),
                    (i1, j1),
                    (i0, j1),
                    (i0 + 1, j0 + 1),
                    (i0 + 1, j1),
                    (i0, j0 + 1),
                ],
            ):
                add(
                    "ring",
                    [ring_node(*p) for p in tri],
                    ring_layer_of_cell[c],
                    ring_axial,
                )

    # Contact gap adjustment for small manufacturing overlaps (snug fit).
    adjusted: dict[str, float] = {}
    bore_gap = ring.inner_radius_m - (rc + a)
    core_gap = rc * math.sin(SECTOR) - a
    for label, value, reference in (
        ("core_ring", bore_gap, ring.inner_radius_m),
        ("core_core", core_gap, a),
    ):
        if value < 0.0:
            if -value > bundle.overlap_tolerance * reference:
                raise CableInputError(
                    f"{label} geometric overlap {-value:.3e} m exceeds the "
                    "snug-fit tolerance; specify an interference model"
                )
            adjusted[label] = -value

    # Filler: blended map between the core arc phi in [0, 5pi/6] and the bore.
    filler_start = 0
    filler_t = 0
    if bundle.filler is not None:
        thickness = [
            math.hypot(
                ring.inner_radius_m * math.cos(thetas[j])
                - (rc + a * math.cos(phis[j])),
                ring.inner_radius_m * math.sin(thetas[j]) - a * math.sin(phis[j]),
            )
            for j in range(0, 2 * n_theta + 1, 2)
        ]
        minimum = 0.25 * min(size, ring_r[-1] - ring_r[0])
        filler_start = next(
            (2 * q for q, h in enumerate(thickness) if h >= minimum), 2 * n_theta
        )
        filler_t = max(2, _cells(max(thickness), size))

        def filler_node(j: int, t: int) -> int:
            s = t / (2 * filler_t)
            cx, cy = rc + a * math.cos(phis[j]), a * math.sin(phis[j])
            bx = ring.inner_radius_m * math.cos(thetas[j])
            by = ring.inner_radius_m * math.sin(thetas[j])
            return nodes.get(
                ("filler", j, t), ((1 - s) * cx + s * bx, (1 - s) * cy + s * by)
            )

        filler = bundle.filler
        for c in range(filler_t):
            t0, t1 = 2 * c, 2 * c + 2
            for q in range(filler_start // 2, n_theta):
                j0, j1 = 2 * q, 2 * q + 2
                for tri in (
                    [
                        (j0, t0),
                        (j1, t0),
                        (j1, t1),
                        (j0 + 1, t0),
                        (j1, t0 + 1),
                        (j0 + 1, t0 + 1),
                    ],
                    [
                        (j0, t0),
                        (j1, t1),
                        (j0, t1),
                        (j0 + 1, t0 + 1),
                        (j0 + 1, t1),
                        (j0, t0 + 1),
                    ],
                ):
                    add("filler", [filler_node(*p) for p in tri], filler, ring_axial)

    xy = np.array(nodes.xy, dtype=np.float64)
    conn = np.array(elements, dtype=np.int64)
    # Orient all triangles counter-clockwise.
    corner = xy[conn[:, :3]]
    area2 = (corner[:, 1, 0] - corner[:, 0, 0]) * (
        corner[:, 2, 1] - corner[:, 0, 1]
    ) - (corner[:, 2, 0] - corner[:, 0, 0]) * (corner[:, 1, 1] - corner[:, 0, 1])
    flip = area2 < 0
    conn[flip] = conn[flip][:, [0, 2, 1, 5, 4, 3]]
    props = np.array(material, dtype=np.float64)
    axial_map = np.array(axial, dtype=np.float64)
    n_nodes = len(xy)
    n_dof = 2 + 2 * n_nodes
    if rho_half[0] == 0.0:
        centre = nodes.keys[("core", 0, 0)]
        centre_dofs = (2 + 2 * centre, 2 + 2 * centre)
    else:
        centre_dofs = (
            2 + 2 * nodes.keys[("core", 0, 0)],
            2 + 2 * nodes.keys[("core", 0, 2 * n_phi)],
        )

    stiffness, load, twist_rigidity = _assemble(
        xy, conn, props, axial_map, centre_dofs, n_dof, groups, rc, cb
    )
    # Grounded stabilisation of in-plane freedoms.
    e_min = float(np.min(props[:, 0]))
    diag = np.zeros(n_dof)
    diag[2:] = bundle.stabilisation * e_min
    diag[1] = twist_rigidity
    stiffness = (stiffness + sp.diags(diag)).tocsr() * SYMMETRY_FACTOR
    load = load * SYMMETRY_FACTOR

    rows: list[dict[int, float]] = []
    gaps: list[float] = []
    families: list[ContactFamily] = []

    def family(name: str, start: int, factor: float) -> None:
        if len(rows) > start:
            families.append(ContactFamily(name, slice(start, len(rows)), factor))

    # Ring bore nodes against the core outer circle (node-to-segment).
    start = len(rows)
    shift = adjusted.get("core_ring", 0.0)
    dphi = phis[2] - phis[0]
    # Where a filler column separates ring and core the filler rows carry the
    # contact; a direct row there would be nearly dependent on them.
    direct_end = filler_start if bundle.filler is not None else 2 * n_theta + 1
    for j in range(direct_end):
        node = ring_node(0, j)
        v = xy[node] - np.array([rc, 0.0])
        dist = float(np.hypot(*v))
        g0 = dist - a + shift
        if g0 > a:
            continue
        normal_c = v / dist
        phi = math.atan2(v[1], v[0])
        q = min(int(phi / dphi), n_phi - 1)
        j0 = 2 * q
        local = (phi - phis[j0]) / dphi
        shape = (
            (1 - local) * (1 - 2 * local),
            4 * local * (1 - local),
            local * (2 * local - 1),
        )
        row: dict[int, float] = {
            2 + 2 * node: float(normal_c[0]),
            3 + 2 * node: float(normal_c[1]),
        }
        for jj, w in zip((j0, j0 + 1, j0 + 2), shape, strict=True):
            m = core_node(core_outer, jj)
            row[2 + 2 * m] = row.get(2 + 2 * m, 0.0) - w * normal_c[0]
            row[3 + 2 * m] = row.get(3 + 2 * m, 0.0) - w * normal_c[1]
        rows.append(row)
        gaps.append(g0)
    family("core_ring", start, 2.0 / SYMMETRY_FACTOR)

    # Core-to-core: contact with the 60 degree mirror plane.
    start = len(rows)
    normal = np.array([-math.sin(SECTOR), math.cos(SECTOR)])
    shift = adjusted.get("core_core", 0.0)
    for j in range(2 * n_phi + 1):
        node = core_node(core_outer, j)
        d0 = -float(xy[node] @ normal) + shift
        if d0 > 0.25 * a:
            continue
        rows.append({2 + 2 * node: -normal[0], 3 + 2 * node: -normal[1]})
        gaps.append(d0)
    family("core_core", start, 1.0 / SYMMETRY_FACTOR)

    if bundle.filler is not None:
        start = len(rows)
        for j in range(filler_start, 2 * n_theta + 1):
            f = nodes.keys[("filler", j, 0)]
            cnode = core_node(core_outer, j)
            n_out = np.array([math.cos(phis[j]), math.sin(phis[j])])
            rows.append(
                {
                    2 + 2 * f: n_out[0],
                    3 + 2 * f: n_out[1],
                    2 + 2 * cnode: -n_out[0],
                    3 + 2 * cnode: -n_out[1],
                }
            )
            gaps.append(0.0)
        family("filler_core", start, 2.0 / SYMMETRY_FACTOR)
        start = len(rows)
        for j in range(filler_start, 2 * n_theta + 1):
            f = nodes.keys[("filler", j, 2 * filler_t)]
            m = ring_node(0, j)
            er = np.array([math.cos(thetas[j]), math.sin(thetas[j])])
            rows.append(
                {
                    2 + 2 * m: er[0],
                    3 + 2 * m: er[1],
                    2 + 2 * f: -er[0],
                    3 + 2 * f: -er[1],
                }
            )
            gaps.append(0.0)
        family("filler_ring", start, 1.0)

    contact = sp.lil_matrix((len(rows), n_dof))
    for i, row in enumerate(rows):
        for k, v in row.items():
            contact[i, k] = v

    # Mean radial displacement of the ring outer surface.
    outer = sp.lil_matrix((1, n_dof))
    i_out = 2 * ring_cells
    span = thetas[-1] - thetas[0]
    for q in range(n_theta):
        j0 = 2 * q
        dtheta = thetas[j0 + 2] - thetas[j0]
        for jj, w in zip((j0, j0 + 1, j0 + 2), (1 / 6, 2 / 3, 1 / 6), strict=True):
            m = ring_node(i_out, jj)
            t = thetas[jj]
            outer[0, 2 + 2 * m] += w * dtheta / span * math.cos(t)
            outer[0, 3 + 2 * m] += w * dtheta / span * math.sin(t)

    transform = _mirror_transform(xy, nodes, n_dof, n_theta)
    core_ea = sum(
        layer.youngs_modulus_Pa
        * math.pi
        * (layer.outer_radius_m**2 - layer.inner_radius_m**2)
        for layer in layers
    )
    core_free = (
        sum(
            layer.youngs_modulus_Pa
            * layer.free_thermal_strain
            * math.pi
            * (layer.outer_radius_m**2 - layer.inner_radius_m**2)
            for layer in layers
        )
        / core_ea
    )
    return TrefoilModel(
        stiffness,
        load,
        contact.tocsr(),
        np.array(gaps, dtype=np.float64),
        tuple(families),
        outer.tocsr(),
        transform,
        xy,
        centre_dofs,
        3 * core_ea,
        core_free,
        beta,
        rc,
        ring_outer,
        adjusted,
        tuple(filled),
        {k: np.array(v, dtype=np.int64) for k, v in groups.items()},
        conn,
        props[:, 0].copy(),
        props[:, 1].copy(),
        props[:, 2].copy(),
        axial_map,
        tuple(owners),
        props[:, 3].copy() if np.any(props[:, 3]) else None,
    )


def _mirror_transform(
    xy: FloatArray,
    nodes: _NodeTable,
    n_dof: int,
    n_theta: int,
) -> sp.csr_matrix:
    """Reduced freedoms: u_y = 0 on y = 0; normal motion zero on 60 degrees.

    Core nodes on the 60 degree line do not exist (the core only touches it).
    Ring and filler nodes on that line slide along it.
    """
    on_axis = np.abs(xy[:, 1]) <= 1e-12 * max(1.0, float(np.max(np.abs(xy))))
    on_sixty = np.zeros(len(xy), dtype=bool)
    last = 2 * n_theta
    for key, index in nodes.keys.items():
        # Keys are ("ring", i, j) and ("filler", j, t).
        if (key[0] == "ring" and key[2] == last) or (
            key[0] == "filler" and key[1] == last
        ):
            on_sixty[index] = True
    rows, cols, vals = [0, 1], [0, 1], [1.0, 1.0]
    column = 2
    tangent = (math.cos(SECTOR), math.sin(SECTOR))
    for n in range(len(xy)):
        dx, dy = 2 + 2 * n, 3 + 2 * n
        if on_axis[n]:
            rows.append(dx)
            cols.append(column)
            vals.append(1.0)
            column += 1
        elif on_sixty[n]:
            rows.extend([dx, dy])
            cols.extend([column, column])
            vals.extend(tangent)
            column += 1
        else:
            rows.extend([dx, dy])
            cols.extend([column, column + 1])
            vals.extend([1.0, 1.0])
            column += 2
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_dof, column))


def _assemble(
    xy: FloatArray,
    conn: NDArray[np.int64],
    props: FloatArray,
    axial_map: FloatArray,
    centre_dofs: tuple[int, int],
    n_dof: int,
    groups: dict[str, list[int]],
    rc: float,
    cb: float,
) -> tuple[sp.coo_matrix, FloatArray, float]:
    """Generalised plane-strain T6 stiffness, free-strain load, twist rigidity."""
    ne = len(conn)
    coords = xy[conn]  # (ne, 6, 2)
    e, nu, e0 = props[:, 0], props[:, 1], props[:, 2]
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    d = np.zeros((ne, 4, 4))
    for i in range(3):
        for j in range(3):
            d[:, i, j] = lam
        d[:, i, i] += 2 * mu
    d[:, 3, 3] = mu
    ndof_e = 16
    ke = np.zeros((ne, ndof_e, ndof_e))
    fe = np.zeros((ne, ndof_e))
    polar = np.zeros(ne)
    core_set = set(groups.get("core", []))
    is_core = np.array([i in core_set for i in range(ne)])
    for (shape, dshape), w in zip(_SHAPES, _QW, strict=True):
        jac = np.einsum("ka,eki->eai", dshape, coords)  # d(x_i)/d(xi_a)
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        if np.any(det <= 0):
            raise CableInputError("degenerate trefoil element; refine the mesh")
        inv = np.empty_like(jac)
        inv[:, 0, 0] = jac[:, 1, 1] / det
        inv[:, 1, 1] = jac[:, 0, 0] / det
        inv[:, 0, 1] = -jac[:, 0, 1] / det
        inv[:, 1, 0] = -jac[:, 1, 0] / det
        dn = np.einsum("ka,eia->eki", dshape, inv)  # dN_k/dx_i
        b = np.zeros((ne, 4, ndof_e))
        b[:, 0, 0:12:2] = dn[:, :, 0]
        b[:, 1, 1:12:2] = dn[:, :, 1]
        b[:, 3, 0:12:2] = dn[:, :, 1]
        b[:, 3, 1:12:2] = dn[:, :, 0]
        b[:, 2, 12:16] = axial_map
        weight = w * det
        ke += np.einsum("e,eai,eab,ebj->eij", weight, b, d, b)
        e0v = np.zeros((ne, 4))
        e0v[:, :3] = e0[:, None]
        e0v[:, :2] += props[:, 3, None]
        fe += np.einsum("e,eai,eab,eb->ei", weight, b, d, e0v)
        point = np.einsum("k,eki->ei", shape, coords)
        r2 = np.where(
            is_core,
            (point[:, 0] - rc) ** 2 + point[:, 1] ** 2,
            point[:, 0] ** 2 + point[:, 1] ** 2,
        )
        polar += weight * mu * r2
    twist = float(np.sum(np.where(is_core, polar * cb**4, polar)))
    dofs = np.zeros((ne, ndof_e), dtype=np.int64)
    dofs[:, 0:12:2] = 2 + 2 * conn
    dofs[:, 1:12:2] = 3 + 2 * conn
    dofs[:, 12] = 0
    dofs[:, 13] = 1
    dofs[:, 14] = np.where(is_core, centre_dofs[0], 0)
    dofs[:, 15] = np.where(is_core, centre_dofs[1], 0)
    # Non-core elements carry zero coefficient on the third extra freedom.
    rows = np.repeat(dofs, ndof_e, axis=1).ravel()
    cols = np.tile(dofs, (1, ndof_e)).ravel()
    k = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))
    load = np.zeros(n_dof)
    np.add.at(load, dofs.ravel(), fe.ravel())
    return k, load, twist
