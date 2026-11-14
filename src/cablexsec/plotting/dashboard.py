# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Exploratory tier: a self-contained HTML *viewer* of result objects.

This is the interactive surface for users — deliberately **separate** from
the evidence-tracked static figures. It is a **viewer, not a second engine**:

    HARD RULE — every number shown comes from a result object (with its
    provenance). The page NEVER recomputes a claimed physics number in
    JavaScript. The Python side serialises existing result-object fields; the JS
    only draws SVG, maps a value to a colour (pure rendering), and shows the
    stored numbers and provenance on demand.

:func:`build_dashboard` writes one standalone HTML file (no network, no external
JS) with the cross-section explorer and result panels (per-wire stress + slip
state, M-κ, capacity, fatigue). Each panel renders its source result's
``provenance`` (method, citations, gates, limitations) verbatim, so the viewer
cannot present a number more confidently than the result allows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cablexsec.plotting.style import COLORS

if TYPE_CHECKING:
    from cablexsec.analysis.capacity import CapacityResult
    from cablexsec.analysis.fatigue import FatigueResult
    from cablexsec.analysis.stress import StressResult
    from cablexsec.api import MkappaResult
    from cablexsec.cables.schema import CableData
    from cablexsec.provenance import Provenance

__all__ = ["build_dashboard"]


def _prov(p: Provenance | None) -> dict[str, Any]:
    """Serialise a provenance record (verbatim — never paraphrased)."""
    if p is None:
        return {"method": "", "citations": [], "gates": [], "limitations": []}
    return {
        "method": p.method,
        "citations": list(p.citations),
        "gates": list(p.gates),
        "limitations": list(p.limitations),
    }


def _cross_section_payload(
    cable: CableData, stress: StressResult | None
) -> dict[str, Any]:
    """Per-wire positions + the result's per-wire NUMBERS (no recomputation)."""
    wires: list[dict[str, Any]] = []
    r_max = 0.0
    for lyr in cable.helical_layers:
        if None in (lyr.n_wires, lyr.r_helix_m, lyr.d_wire_m):
            continue
        assert lyr.n_wires and lyr.r_helix_m and lyr.d_wire_m
        r_h, r_w = lyr.r_helix_m, lyr.d_wire_m / 2.0
        r_max = max(r_max, r_h + r_w)
        has = stress is not None and lyr.name in stress.phi_deg
        if has:
            assert stress is not None
            phis = [math.radians(d) for d in stress.phi_deg[lyr.name]]
        else:
            phis = [2 * math.pi * k / lyr.n_wires for k in range(lyr.n_wires)]
        for k, phi in enumerate(phis):
            w: dict[str, Any] = {
                "layer": lyr.name,
                "x": r_h * math.cos(phi),
                "y": r_h * math.sin(phi),
                "r": r_w,
                "phi_deg": math.degrees(phi),
            }
            if has:
                assert stress is not None
                # every value copied straight from the result object
                w.update(
                    sigma_total_MPa=float(stress.sigma_total_Pa[lyr.name][k]) / 1e6,
                    sigma_A_MPa=float(stress.sigma_A_Pa[lyr.name][k]) / 1e6,
                    sigma_B_MPa=float(stress.sigma_B_Pa[lyr.name][k]) / 1e6,
                    sigma_f_MPa=float(stress.sigma_f_Pa[lyr.name][k]) / 1e6,
                    sigma_vm_MPa=float(stress.sigma_vm_Pa[lyr.name][k]) / 1e6,
                    slipping=bool(stress.slipping[lyr.name][k]),
                    kappa_c=float(stress.kappa_c_1_m[lyr.name][k]),
                )
            wires.append(w)
    return {"wires": wires, "r_max": r_max, "has_stress": stress is not None}


def build_dashboard(
    cable: CableData,
    *,
    stress: StressResult | None = None,
    mkappa: MkappaResult | None = None,
    capacity: CapacityResult | None = None,
    fatigue: FatigueResult | None = None,
    out_path: str | Path = "cablexsec_dashboard.html",
    title: str = "",
) -> Path:
    """Write a standalone HTML viewer of the supplied result objects.

    Parameters
    ----------
    cable : CableData
        The cable (for the cross-section geometry).
    stress : StressResult, optional
        Per-wire stress to drive the cross-section explorer.
    mkappa, capacity, fatigue : optional
        Result objects rendered as summary panels (numbers + provenance).
    out_path : str or pathlib.Path, optional
        Output HTML path.
    title : str, optional

    Returns
    -------
    pathlib.Path
        The written HTML file.
    """
    panels: list[dict[str, Any]] = []
    if mkappa is not None:
        panels.append(
            {
                "name": "Moment–curvature",
                "rows": [
                    ("W_diss [J/m]", f"{mkappa.W_diss_J_per_m:.1f}"),
                    ("EI_apparent [kN·m²]", f"{mkappa.EI_apparent_Nm2 / 1e3:.1f}"),
                    ("EI_max [kN·m²]", f"{mkappa.EI_max_Nm2 / 1e3:.1f}"),
                    ("EI_min [kN·m²]", f"{mkappa.EI_min_Nm2 / 1e3:.2f}"),
                ],
                "prov": _prov(mkappa.provenance),
            }
        )
    if capacity is not None:
        i_peak = int(
            max(
                range(len(capacity.kappa_max_1_m)),
                key=lambda j: capacity.kappa_max_1_m[j],
            )
        )
        panels.append(
            {
                "name": "Capacity envelope",
                "rows": [
                    (
                        "peak κ_max [1/m]",
                        f"{float(capacity.kappa_max_1_m[i_peak]):.4f}",
                    ),
                    ("governing @ peak", str(capacity.governing[i_peak])),
                    (
                        "max UF bird-caging",
                        f"{float(max(capacity.uf_bird_caging)):.3f}",
                    ),
                    ("max UF conductor", f"{float(max(capacity.uf_conductor)):.3f}"),
                ],
                "prov": _prov(capacity.provenance),
            }
        )
    if fatigue is not None:
        panels.append(
            {
                "name": "Fatigue",
                "rows": [
                    ("critical layer", str(fatigue.critical_layer)),
                    ("critical φ [deg]", f"{fatigue.phi_critical_deg:.0f}"),
                    ("governing", str(fatigue.governing)),
                    ("fretting fraction", f"{fatigue.D_fretting_fraction:.3f}"),
                ],
                "prov": _prov(fatigue.provenance),
            }
        )

    payload = {
        "title": title or f"{cable.cable.name} — CableXSec viewer",
        "cross_section": _cross_section_payload(cable, stress),
        "stress_prov": _prov(stress.provenance if stress else None),
        "panels": panels,
        "palette": {k: COLORS[k] for k in COLORS},
    }
    html = _HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, allow_nan=False))
    p = Path(out_path)
    p.write_text(html, encoding="utf-8")
    return p


# The viewer is render-only: it draws SVG, maps value→colour, and shows stored
# numbers + provenance. It contains NO physics formula. data-viewer="render-only"
# is the contract marker the L7.2 test checks for.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-viewer="render-only">
<head>
<meta charset="utf-8"/>
<title>CableXSec viewer</title>
<style>
  body { font-family: "Times New Roman", serif; color:#000; margin:0; background:#fff; }
  header { background:#000; color:#fff; padding:10px 16px; }
  header h1 { font-size:18px; margin:0; }
  .banner { background:#CC6600; color:#fff; padding:6px 16px; font-size:13px; }
  .wrap { display:flex; flex-wrap:wrap; gap:18px; padding:16px; }
  .card { border:1px solid #666; border-radius:4px; padding:12px; }
  svg { background:#fff; }
  circle.wire { stroke:#000; stroke-width:0.4; cursor:pointer; }
  table { border-collapse:collapse; font-size:13px; }
  td,th { border:1px solid #999; padding:3px 8px; text-align:left; }
  .prov { font-size:11px; color:#666; max-width:380px; margin-top:8px; }
  .prov b { color:#000; }
  .lim { color:#CC0000; }
  #detail { font-size:13px; min-width:200px; }
  .legend { font-size:11px; color:#666; }
</style>
</head>
<body>
<header><h1 id="ttl"></h1></header>
<div class="banner">VIEWER — displays result-object values and their provenance.
It computes no claimed number; every figure here comes from a CableXSec result.</div>
<div class="wrap">
  <div class="card">
    <div id="csTitle"><b>Cross-section</b></div>
    <svg id="cs" width="420" height="420"></svg>
    <div class="legend" id="csLegend"></div>
  </div>
  <div class="card" id="detail"><b>Wire detail</b><div id="detailBody">
    Click a wire.</div>
    <div class="prov" id="stressProv"></div>
  </div>
  <div class="card" id="panels"></div>
</div>
<script>
const DATA = /*__DATA__*/;
document.getElementById("ttl").textContent = DATA.title;

// value -> colour: pure rendering (blue-white-red), NOT a physics computation.
function divColor(v, vmax){
  if(!vmax) return "#FFFFFF";
  const t = Math.max(-1, Math.min(1, v/vmax));
  const L=(a,b,u)=>Math.round(a+(b-a)*u);
  if(t<0){const u=1+t;
    return `rgb(${L(0,255,u)},${L(0,255,u)},${L(204,255,u)})`;}
  const u=1-t;
  return `rgb(${L(204,255,u)},${L(0,255,u)},${L(0,255,u)})`;
}
function provHTML(p){
  if(!p||!p.method) return "";
  let h=`<b>method:</b> ${p.method}<br>`;
  if(p.citations.length) h+=`<b>cite:</b> ${p.citations.join("; ")}<br>`;
  if(p.gates.length) h+=`<b>gates:</b> ${p.gates.join("; ")}<br>`;
  if(p.limitations.length) h+=`<b>limitations:</b><br>`+
    p.limitations.map(l=>`<span class="lim">• ${l}</span>`).join("<br>");
  return h;
}

const cs=DATA.cross_section, svg=document.getElementById("cs");
const W=420, R=DATA.cross_section.r_max||1, sc=190/R, cx=210, cy=210;
let vmax=0;
if(cs.has_stress) for(const w of cs.wires)
  vmax=Math.max(vmax, Math.abs(w.sigma_total_MPa||0));
for(const w of cs.wires){
  const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
  c.setAttribute("class","wire");
  c.setAttribute("cx", cx+w.x*sc); c.setAttribute("cy", cy-w.y*sc);
  c.setAttribute("r", Math.max(2, w.r*sc));
  const fill = cs.has_stress ? divColor(w.sigma_total_MPa, vmax) : "#666666";
  if(cs.has_stress && w.slipping){ c.setAttribute("fill","none");
    c.setAttribute("stroke", fill); c.setAttribute("stroke-width","1.4"); }
  else c.setAttribute("fill", fill);
  c.onclick=()=>showWire(w);
  svg.appendChild(c);
}
document.getElementById("csLegend").textContent = cs.has_stress
  ? `colour = σ_total (blue −${vmax.toFixed(0)} … white 0 … `
    + `red +${vmax.toFixed(0)} MPa); open = slipping`
  : "colour = material (geometry only)";

function showWire(w){
  let h=`<table><tr><th>field</th><th>value (from result)</th></tr>`+
    `<tr><td>layer</td><td>${w.layer}</td></tr>`+
    `<tr><td>φ [deg]</td><td>${w.phi_deg.toFixed(1)}</td></tr>`;
  if("sigma_total_MPa" in w){
    h+=`<tr><td>σ_total [MPa]</td><td>${w.sigma_total_MPa.toFixed(2)}</td></tr>`+
       `<tr><td>σ_A [MPa]</td><td>${w.sigma_A_MPa.toFixed(2)}</td></tr>`+
       `<tr><td>σ_B [MPa]</td><td>${w.sigma_B_MPa.toFixed(2)}</td></tr>`+
       `<tr><td>σ_f [MPa]</td><td>${w.sigma_f_MPa.toFixed(2)}</td></tr>`+
       `<tr><td>Axial/contact indicator [MPa]</td>`+
       `<td>${w.sigma_vm_MPa.toFixed(2)}</td></tr>`+
       `<tr><td>slipping</td><td>${w.slipping}</td></tr>`+
       `<tr><td>κ_c [1/m]</td><td>${w.kappa_c.toFixed(5)}</td></tr>`;
  }
  h+=`</table>`;
  document.getElementById("detailBody").innerHTML=h;
}
document.getElementById("stressProv").innerHTML = provHTML(DATA.stress_prov);

const pc=document.getElementById("panels");
for(const pn of DATA.panels){
  let h=`<b>${pn.name}</b><table>`;
  for(const r of pn.rows) h+=`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`;
  h+=`</table><div class="prov">${provHTML(pn.prov)}</div>`;
  const d=document.createElement("div"); d.innerHTML=h;
  d.style.marginBottom="14px"; pc.appendChild(d);
}
</script>
</body>
</html>
"""
