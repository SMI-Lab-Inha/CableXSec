# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Bending test of a finite specimen with slip redistribution along the wires.

:func:`~cablexsec.analysis.beam_test.simulate_bending_test` treats every
station of a specimen as an independent section. A real specimen is one to a
few lay lengths long between supports, grips or end fittings, so a wire
passes through stations of different curvature and its axial force is set by
friction along its own helix, between ends that are anchored or free. This
module couples the statically determinate beam of the test with the
along-wire model of :mod:`cablexsec.mechanics.along_wire` for every wire of
every helical layer:

* station moment ``M(x) = EI_el kappa(x) + sum_layers g_0 R cos a
  sum_q n_q(x) cos theta_q(x)``, with ``n_q`` the along-wire force of wire
  ``q`` interpolated to the station and ``EI_el`` the elastic (full-slip)
  stiffness of everything else;
* equilibrium ``M(x, t) = F(t) m(x)`` at every station (``m = 0`` on
  overhangs beyond the supports), plus the control equation;
* every load step is solved by a Newton-type iteration whose station tangent
  is the row sum of the along-wire Jacobian (the linear response of the
  wires on their converged stick/slip set to a uniform unit curvature),
  with backtracking on the equilibrium residual and step halving on
  failure.

The infinite-length limit is the distributed sectional law
(``friction_model="distributed"`` of the section) with the ``"spring"``
stick convention; :class:`AlongWireSpec` sets the stick convention, the
capstan term, the wire ends and the overhangs explicitly.

Limits: Euler--Bernoulli beam on frictionless supports (no shear deformation,
P-delta or self-weight), planar bending, axial wire slip only, contact
pressure held at the section state (no bending-induced redistribution), no
laid-core friction, one slipping order per conductor ring. The elastic part
and the chain moment are evaluated at the stations; the chains carry the
curvature linearly interpolated between stations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cablexsec.exceptions import CableInputError, CableSolverError
from cablexsec.mechanics.along_wire import (
    END_CONDITIONS,
    STICK_CONVENTIONS,
    WireChain,
    WireChainLayer,
    chain_layer_jacobian,
    chain_layer_step,
)
from cablexsec.provenance import require_finite

if TYPE_CHECKING:
    from cablexsec.analysis.beam_test import BendingProtocol, _Beam
    from cablexsec.mechanics.strand_slip import ConductorRing, StrandSlipParameters

__all__ = [
    "AlongWireOutput",
    "AlongWireSection",
    "AlongWireSpec",
]

FloatArray = NDArray[np.float64]
_State = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
_Trial = tuple[_State, FloatArray, FloatArray, FloatArray]


@dataclass(frozen=True)
class AlongWireSpec:
    """Along-wire option of :func:`~cablexsec.analysis.beam_test.simulate_bending_test`.

    Attributes
    ----------
    ends : (str, str)
        Wire end condition at the left and right specimen ends, ``"anchored"``
        (end fitting, grip, clamping fixture) or ``"free"`` (cut end).
    overhang_m : (float, float)
        Straight length of specimen beyond the left and right supports [m]
        (zero moment); the wire ends are at ``-overhang[0]`` and
        ``L + overhang[1]``.
    stick : {"spring", "rigid", "scaled"}
        Stick-stiffness convention (:mod:`cablexsec.mechanics.along_wire`).
        Used when the section is built from a cable.
    capstan : bool
        Capstan term of the friction bound (the package cap); built from a
        cable only.
    segments_per_pitch : int
        Wire segments per lay length.
    sense : int
        Sense in which the wires turn along ``+x`` (planar bending results do
        not depend on it at stations in the bending plane).
    substeps : int
        Uniform subdivision of every protocol interval.
    max_iterations : int
        Newton-type iterations per step before the step is halved.
    """

    ends: tuple[str, str] = ("free", "free")
    overhang_m: tuple[float, float] = (0.0, 0.0)
    stick: str = "spring"
    capstan: bool = True
    segments_per_pitch: int = 32
    sense: int = 1
    substeps: int = 1
    max_iterations: int = 60

    def __post_init__(self) -> None:
        ends = tuple(self.ends)
        if len(ends) != 2 or any(e not in END_CONDITIONS for e in ends):
            raise CableInputError(f"ends must be two of {END_CONDITIONS}")
        object.__setattr__(self, "ends", (str(ends[0]), str(ends[1])))
        over = tuple(float(v) for v in self.overhang_m)
        if len(over) != 2 or any(not math.isfinite(v) or v < 0 for v in over):
            raise CableInputError("overhang_m must be two non-negative lengths")
        object.__setattr__(self, "overhang_m", over)
        if self.stick not in STICK_CONVENTIONS:
            raise CableInputError(f"stick must be one of {STICK_CONVENTIONS}")
        if self.sense not in (1, -1):
            raise CableInputError("sense must be +1 or -1")
        for name, low in (
            ("segments_per_pitch", 8),
            ("substeps", 1),
            ("max_iterations", 4),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < low
            ):
                raise CableInputError(f"{name} must be an integer of at least {low}")


@dataclass(frozen=True)
class AlongWireSection:
    """Elastic stiffness plus the helical layers solved along their wires."""

    layers: tuple[WireChainLayer, ...]
    ei_elastic_Nm2: float
    description: str = ""

    def __post_init__(self) -> None:
        require_finite(self.ei_elastic_Nm2, "ei_elastic_Nm2")
        if self.ei_elastic_Nm2 <= 0:
            raise CableInputError("ei_elastic_Nm2 must be positive")
        if not all(isinstance(lay, WireChainLayer) for lay in self.layers):
            raise CableInputError("layers must be WireChainLayer instances")
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def ei_stick_Nm2(self) -> float:
        """Far-field stick stiffness (elastic part plus every layer in stick)."""
        total = self.ei_elastic_Nm2
        for lay in self.layers:
            far = lay.k_hat / (1.0 + lay.k_hat)
            total += 0.5 * lay.n_wires * lay.lever_Nm * far / lay.kappa_c_1_m
        return total

    @classmethod
    def from_cable(
        cls,
        cable: object,
        *,
        stick: str = "spring",
        capstan: bool = True,
        sense: int = 1,
        tension_N: float = 0.0,
        mu: float | None = None,
        contact_model: str | None = None,
        section_state: object | None = None,
        radial_state: object | None = None,
    ) -> AlongWireSection:
        """Armour layers and full-slip stiffness of a cable (the
        :func:`~cablexsec.api.compute_mkappa` section without core friction)."""
        from cablexsec.analysis.beam_test import (  # noqa: PLC0415
            _mkappa_law_for_along_wire,
        )

        law = _mkappa_law_for_along_wire(
            cable,
            tension_N=tension_N,
            mu=mu,
            contact_model=contact_model,
            section_state=section_state,
            radial_state=radial_state,
        )
        layers = tuple(
            WireChainLayer.from_layer_spec(
                spec, stick=stick, capstan=capstan, sense=sense, name=f"layer {i}"
            )
            for i, spec in enumerate(law.layers)
            if spec.mu > 0 and spec.contact_pressure_Pa > 0
        )
        inert = [
            spec
            for spec in law.layers
            if not (spec.mu > 0 and spec.contact_pressure_Pa > 0)
        ]
        if inert:
            raise CableInputError(
                "an armour layer without friction cannot be solved along its wires"
            )
        return cls(
            layers,
            float(law.ei_min),
            f"cable armour along the wires ({stick} stick, capstan "
            f"{'on' if capstan else 'off'})",
        )

    @classmethod
    def from_conductor(
        cls,
        rings: Sequence[ConductorRing],
        E_Pa: float,
        parameters: Sequence[StrandSlipParameters],
        *,
        tension_N: float = 0.0,
        additional_EI_Nm2: float = 0.0,
    ) -> AlongWireSection:
        """Conductor rings along their wires (Papailiou cap shape, rigid stick).

        The elastic part is ``additional_EI_Nm2`` plus the wire bending
        stiffness and the slip-slope part of every ring, as in
        :meth:`~cablexsec.analysis.beam_test.SectionLaw.from_conductor`.
        """
        from cablexsec.mechanics.core_lay import bending_factor  # noqa: PLC0415

        elastic = float(additional_EI_Nm2)
        layers = []
        for i, (ring, p) in enumerate(zip(rings, parameters, strict=True)):
            elastic += (
                E_Pa
                * ring.n_wires
                * math.pi
                * ring.d_wire_m**4
                / 64.0
                * bending_factor(ring.lay_angle_rad)
            )
            if ring.r_helix_m == 0:
                continue
            lever = (
                0.5
                * ring.n_wires
                * E_Pa
                * ring.area_m2
                * math.cos(ring.lay_angle_rad)
                * ring.r_helix_m
            )
            elastic += lever * float(p.beta_m[-1])
            layers.append(
                WireChainLayer.from_conductor_ring(
                    ring,
                    E_Pa,
                    p,
                    tension_N=tension_N,
                    name=f"ring {i}",
                )
            )
        return cls(tuple(layers), elastic, "conductor rings along the wires")


@dataclass(frozen=True)
class AlongWireOutput:
    """Along-wire details of a test (all stations, overhangs included).

    ``x_m`` are all stations, ``span`` the slice of the span stations in
    them; ``curvature_1_m``, ``moment_Nm`` and ``armour_moment_Nm`` (layers,
    stations, samples) are station fields; ``energy_J`` holds the cumulative
    ``[W_be, D, U, R]`` of all wires (J); ``iterations`` per sample the
    Newton-type iterations; ``residual`` per sample the equilibrium residual
    relative to the moment scale.
    """

    x_m: FloatArray
    span: slice
    curvature_1_m: FloatArray
    moment_Nm: FloatArray
    armour_moment_Nm: FloatArray
    energy_J: FloatArray
    iterations: NDArray[np.int64]
    residual: FloatArray
    segments: tuple[int, ...]
    spec: AlongWireSpec
    notes: tuple[str, ...] = field(default=())


class _LayerState:
    """Chain, interpolation and committed state of one layer."""

    def __init__(self, chain: WireChain, x_all: FloatArray) -> None:
        self.chain = chain
        lay = chain.layer
        self.lay = lay
        pos = np.searchsorted(x_all, chain.x_seg_m, side="right") - 1
        pos = np.clip(pos, 0, x_all.size - 2)
        frac = (chain.x_seg_m - x_all[pos]) / (x_all[pos + 1] - x_all[pos])
        self.i0 = pos
        self.f = np.clip(frac, 0.0, 1.0)
        self.j0, self.fj = chain.record_weights(x_all)
        q = np.arange(lay.n_wires, dtype=np.float64)
        self.cos_st = np.cos(lay.azimuth(x_all[None, :], q[:, None]))
        n_w, m = lay.n_wires, chain.segments
        self.state = (
            np.zeros((n_w, m + 1)),
            np.zeros((n_w, m + 1)),
            np.zeros((n_w, m)),
            np.zeros((n_w, m + 1)),
        )
        self.e = np.zeros((n_w, m))
        self.guess = np.zeros((n_w, m + 1))
        # node -> station bins for the dissipation field
        edges = np.r_[x_all[0], 0.5 * (x_all[1:] + x_all[:-1]), x_all[-1]]
        self.node_bin = np.clip(
            np.searchsorted(edges, chain.x_node_m, side="right") - 1, 0, x_all.size - 1
        )

    def strain(self, kappa: FloatArray) -> FloatArray:
        k_seg = kappa[self.i0] * (1.0 - self.f) + kappa[self.i0 + 1] * self.f
        return np.ascontiguousarray(self.chain.cos_seg * (k_seg / self.lay.kappa_c_1_m))

    def at_stations(self, n: FloatArray) -> FloatArray:
        vals = n[:, self.j0] * (1.0 - self.fj) + n[:, self.j0 + 1] * self.fj
        return np.asarray(self.lay.lever_Nm * np.sum(self.cos_st * vals, axis=0))

    def trial(
        self, kappa: FloatArray, jacobian: bool
    ) -> tuple[_State, FloatArray, FloatArray, FloatArray | None, FloatArray]:
        e_new = self.strain(kappa)
        new, energy = chain_layer_step(
            self.chain, e_new, self.e, 1.0, self.state, self.guess
        )
        moment = self.at_stations(new[2])
        jac = (
            chain_layer_jacobian(
                self.chain,
                self.state,
                self.guess,
                new[2],
                1.0,
                self.i0,
                self.f,
                self.j0,
                self.fj,
                self.cos_st,
            )
            if jacobian
            else None
        )
        return new, energy, moment, jac, e_new

    def dissipation_nodes(self, new: _State) -> FloatArray:
        u0, z0 = self.state[0], self.state[1]
        u1, z1, _n1, t1 = new
        d = self.chain.weights[None, :] * t1 * ((u1 - u0) - (z1 - z0))
        return np.asarray(np.sum(d, axis=0) * self.lay.energy_unit_J)

    def stored(self, state: _State) -> tuple[float, FloatArray]:
        _u, z, n, _t = state
        lay = self.lay
        h = self.chain.h
        seg = 0.5 * h * np.sum(n * n, axis=0)
        node = np.where(
            self.chain.fixed,
            0.0,
            0.5 * self.chain.weights * lay.k_hat * np.sum(z * z, axis=0),
        )
        total = (float(np.sum(seg)) + float(np.sum(node))) * lay.energy_unit_J
        return total, np.asarray(node * lay.energy_unit_J)

    def commit(self, new: _State, e_new: FloatArray) -> None:
        self.state = (new[0], new[1], new[2], new[3])
        self.e = e_new


def run_along_wire_test(
    section: AlongWireSection,
    beam: _Beam,
    protocol: BendingProtocol,
    spec: AlongWireSpec,
    *,
    definition: str,
    gauges: FloatArray,
    stations_per_segment: int,
    rtol: float,
) -> dict[str, object]:
    """Coupled beam--wire simulation (called by ``simulate_bending_test``).

    Returns the station fields and reductions as a dictionary that the caller
    turns into a :class:`~cablexsec.analysis.beam_test.BendingTestResult`.
    """
    from cablexsec.analysis.beam_test import (  # noqa: PLC0415
        _product_weights,
        _trapezoid_weights,
    )

    length = float(beam.length)
    keys = sorted(set(beam.key_points))
    n_seg = int(stations_per_segment)
    x_span = np.unique(
        np.concatenate(
            [
                np.linspace(a, b, n_seg + 1)
                for a, b in zip(keys[:-1], keys[1:], strict=True)
            ]
            + [gauges]
        )
    )
    dx = length / (n_seg * (len(keys) - 1))
    oh_l, oh_r = spec.overhang_m
    left = (
        np.linspace(-oh_l, 0.0, max(1, int(math.ceil(oh_l / dx))) + 1)[:-1]
        if oh_l > 0
        else np.zeros(0)
    )
    right = (
        np.linspace(length, length + oh_r, max(1, int(math.ceil(oh_r / dx))) + 1)[1:]
        if oh_r > 0
        else np.zeros(0)
    )
    x_all = np.r_[left, x_span, right]
    span = slice(left.size, left.size + x_span.size)
    n_s = x_all.size
    mbar = np.zeros(n_s)
    mbar[span] = beam.influence(x_span)
    i_meas_span = int(np.argmin(np.abs(x_span - beam.measuring_x)))
    i_meas = left.size + i_meas_span
    mbar_meas = float(mbar[i_meas])

    def embed(wv: FloatArray) -> FloatArray:
        out = np.zeros(n_s)
        out[span] = wv
        return out

    disp_w = embed(_product_weights(x_span, beam.influence(x_span)))
    gauge_w = np.array(
        [embed(_product_weights(x_span, beam.kernel(x_span, g))) for g in gauges]
    ).reshape(gauges.size, n_s)
    if definition == "station":
        curv_w = np.zeros(n_s)
        curv_w[i_meas] = 1.0
    elif definition == "elastic":
        curv_w = beam.elastic_factor * embed(
            _product_weights(x_span, beam.kernel(x_span, beam.elastic_x))
        )
    else:
        vand = np.column_stack((gauges**2, gauges, np.ones_like(gauges)))
        curv_w = -2.0 * (np.linalg.pinv(vand)[0] @ gauge_w)
    q_w = _trapezoid_weights(x_all)

    states = [
        _LayerState(
            WireChain.build(
                lay,
                -oh_l,
                length + oh_r,
                ends=spec.ends,
                segments_per_pitch=spec.segments_per_pitch,
            ),
            x_all,
        )
        for lay in section.layers
    ]
    ei = section.ei_elastic_Nm2

    values = np.asarray(protocol.values, dtype=np.float64)
    control = str(protocol.control)
    n_t = values.size
    if control == "curvature":
        weights_c: FloatArray | None = curv_w
    elif control == "displacement":
        weights_c = disp_w
    else:
        weights_c = None
    if weights_c is not None and float(weights_c @ (mbar / ei)) <= 0:
        raise CableInputError(
            "the controlled quantity does not increase with the load for this "
            "arrangement and curvature definition"
        )

    kappa = np.zeros(n_s)
    load = 0.0
    m_scale = 0.0
    out_k = np.zeros((n_s, n_t))
    out_m = np.zeros((n_s, n_t))
    out_arm = np.zeros((len(states), n_s, n_t))
    out_f = np.zeros(n_t)
    energy = np.zeros(4)
    out_energy = np.zeros((n_t, 4))
    fr_field = np.zeros(n_s)
    out_fr = np.zeros((n_s, n_t))
    out_st = np.zeros((n_s, n_t))
    iterations = np.zeros(n_t, dtype=np.int64)
    residuals = np.zeros(n_t)
    evaluations = 0

    def evaluate(kap: FloatArray) -> tuple[FloatArray, list[_Trial]]:
        nonlocal evaluations
        evaluations += 1
        moment = ei * kap
        trials: list[_Trial] = []
        for st in states:
            new, en, mom, _jac, e_new = st.trial(kap, False)
            moment = moment + mom
            trials.append((new, en, mom, e_new))
        return moment, trials

    def jacobian(trials: list[_Trial]) -> FloatArray:
        jac: FloatArray = np.diag(np.full(n_s, float(ei)))
        for st, (new, _en, _mom, _e) in zip(states, trials, strict=True):
            jac += chain_layer_jacobian(
                st.chain,
                st.state,
                st.guess,
                new[2],
                1.0,
                st.i0,
                st.f,
                st.j0,
                st.fj,
                st.cos_st,
            )
        return jac

    def solve_step(
        target: float,
    ) -> tuple[int, float, list[_Trial], FloatArray, float, FloatArray]:
        """Equilibrium at one control value from the committed state (Newton on
        the station curvatures and the load, Jacobian of the converged
        stick/slip set, backtracking on the residual)."""
        kap = kappa.copy()
        f = load
        last_res = math.inf
        last_kap = kap.copy()
        last_f = f
        step_k = np.zeros(n_s)
        step_f = 0.0
        backtracks = 0
        rises = 0
        for it in range(spec.max_iterations):
            moment, trials = evaluate(kap)
            r = f * mbar - moment
            scale = max(
                m_scale, float(np.max(np.abs(moment))), float(np.max(np.abs(f * mbar)))
            )
            res = float(np.max(np.abs(r))) / scale if scale > 0 else 0.0
            if weights_c is None:
                ctrl = f - target if control == "load" else f * mbar_meas - target
                c_scale = max(
                    abs(target),
                    abs(f * (mbar_meas if control == "moment" else 1.0)),
                    1e-300,
                )
            else:
                ctrl = float(weights_c @ kap) - target
                c_scale = max(abs(target), float(np.max(np.abs(values))), 1e-300)
            if res <= rtol and abs(ctrl) <= rtol * c_scale:
                return it, res, trials, kap, f, moment
            merit = max(res, abs(ctrl) / c_scale)
            if merit > last_res:
                rises += 1
            else:
                rises = 0
            if rises >= 3 and backtracks < 8:
                # several plain semismooth steps without progress: damp
                backtracks += 1
                step_k *= 0.5
                step_f *= 0.5
                kap = last_kap + step_k
                f = last_f + step_f
                continue
            last_res = merit
            last_kap = kap.copy()
            last_f = f
            jac = jacobian(trials)
            try:
                sol = np.linalg.solve(jac, np.column_stack((r, mbar)))
            except np.linalg.LinAlgError as exc:
                raise CableSolverError("singular along-wire beam Jacobian") from exc
            u_r, u_m = sol[:, 0], sol[:, 1]
            if weights_c is None:
                f_target = target if control == "load" else target / mbar_meas
                d_f = f_target - f
            else:
                denom = float(weights_c @ u_m)
                if denom == 0 or not math.isfinite(denom):
                    raise CableSolverError("the control is not reached by the load")
                d_f = (target - float(weights_c @ kap) - float(weights_c @ u_r)) / denom
            step_k = u_r + d_f * u_m
            step_f = d_f
            kap = kap + step_k
            f = f + step_f
        raise CableSolverError("along-wire bending step did not converge")

    def advance(c0: float, c1: float, depth: int) -> tuple[int, float]:
        nonlocal kappa, load, energy, m_scale
        try:
            it, res, trials, kap, f, moment = solve_step(c1)
        except (CableSolverError, CableInputError):
            if depth >= 8:
                raise
            mid = 0.5 * (c0 + c1)
            i1, r1 = advance(c0, mid, depth + 1)
            i2, r2 = advance(mid, c1, depth + 1)
            return i1 + i2, max(r1, r2)
        # commit
        w_be = diss = rem = 0.0
        for st, (new, en, _mom, e_new) in zip(states, trials, strict=True):
            fr_field[:] += np.bincount(
                st.node_bin,
                weights=st.dissipation_nodes(new),
                minlength=n_s,
            )
            en_a = np.asarray(en) * st.lay.energy_unit_J
            w_be += float(en_a[:, 0].sum())
            diss += float(en_a[:, 1].sum())
            rem += float(en_a[:, 2].sum())
            st.commit(new, e_new)
            st.guess[:] = 0.0
        energy[0] += w_be
        energy[1] += diss
        energy[3] += rem
        kappa = kap
        load = f
        m_scale = max(m_scale, float(np.max(np.abs(moment))))
        return it, res

    previous = 0.0
    for k in range(n_t):
        target = float(values[k])
        start = previous
        its = 0
        worst = 0.0
        for s in range(1, spec.substeps + 1):
            c0 = start + (target - start) * (s - 1) / spec.substeps
            c1 = start + (target - start) * s / spec.substeps
            i, r = advance(c0, c1, 0)
            its += i
            worst = max(worst, r)
        previous = target
        iterations[k] = its
        residuals[k] = worst
        moment = ei * kappa
        stored_total = 0.0
        node_stored = np.zeros(n_s)
        for j, st in enumerate(states):
            arm = st.at_stations(st.state[2])
            out_arm[j, :, k] = arm
            moment = moment + arm
            tot, nodes = st.stored(st.state)
            stored_total += tot
            node_stored += np.bincount(st.node_bin, weights=nodes, minlength=n_s)
            seg_n = st.state[2]
            seg_e = (
                0.5 * st.chain.h * np.sum(seg_n * seg_n, axis=0) * st.lay.energy_unit_J
            )
            seg_bin = np.clip(
                np.searchsorted(
                    np.r_[x_all[0], 0.5 * (x_all[1:] + x_all[:-1]), x_all[-1]],
                    st.chain.x_seg_m,
                    side="right",
                )
                - 1,
                0,
                n_s - 1,
            )
            node_stored += np.bincount(seg_bin, weights=seg_e, minlength=n_s)
        elastic = 0.5 * ei * kappa**2
        elastic_total = float(q_w @ elastic)
        energy[2] = stored_total + elastic_total
        out_energy[k] = energy
        out_energy[k, 0] += elastic_total  # internal work: chains plus elastic part
        out_k[:, k] = kappa
        out_m[:, k] = moment
        out_f[k] = load
        out_fr[:, k] = fr_field / q_w
        out_st[:, k] = node_stored / q_w + elastic
    return {
        "x_all": x_all,
        "span": span,
        "x_span": x_span,
        "mbar": mbar,
        "i_meas": i_meas,
        "mbar_meas": mbar_meas,
        "disp_w": disp_w,
        "curv_w": curv_w,
        "gauge_w": gauge_w,
        "q_w": q_w,
        "kappa": out_k,
        "moment": out_m,
        "armour": out_arm,
        "load": out_f,
        "energy": out_energy,
        "friction_field": out_fr,
        "stored_field": out_st,
        "iterations": iterations,
        "residual": residuals,
        "evaluations": evaluations,
        "segments": tuple(st.chain.segments for st in states),
    }
