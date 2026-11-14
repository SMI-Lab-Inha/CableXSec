.. SPDX-License-Identifier: Apache-2.0

CableXSec
=========

**CableXSec** is an open-source (`Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_)
Python library for the local (cross-section) mechanics of dynamic power cables. From a cable
definition and a section load history it computes the axial, torsional and bending stiffness,
the hysteretic moment–curvature response of helical armour with frictional slip, the stresses
of every wire and component at fixed material points, contact pressures and slip, temperatures,
and fatigue damage. It reads the results of global analyses (CableDyn, RIFLEX, CSV exports) and
turns them into the fatigue damage of the armour, the conductor and the sheaths.

.. admonition:: Release status
   :class: note

   CableXSec ``v0.1.0`` is the first public release. It is distributed as a source
   repository and a pure-Python wheel on the
   `GitHub release page <https://github.com/SMI-Lab-Inha/CableXSec/releases>`_. Every
   comparison with published data, including the comparisons that do not meet their
   targets, is listed in :doc:`validation`. The project remains below ``1.0.0``, so public
   interfaces may change between minor releases.

What CableXSec computes
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Area
     - Calculations
   * - Section properties
     - axial, torsional and coupling stiffness; full-slip and no-slip bending bounds;
       plane-section and shear-compliant stick; load sharing; neutral axes of offset cores
   * - Contact state
     - solved radial chain of every layer, winding (manufacturing) clamping, tension,
       pressure and thermal contributions, sealed and flooded annuli
   * - Bending
     - hysteretic moment–curvature loops with the neutral-axis slip front, Coulomb
       saturation, distributed and laid-core friction, stored and dissipated work
   * - Stress
     - axial, local bending and frictional stress of every wire at fixed material points;
       conductor strands, screens, sheaths and tapes; transverse slip
   * - Contact mechanics
     - Hertz contact and subsurface stress, Mindlin partial slip, fretting regimes, wear,
       SWT critical-plane recovery
   * - Thermal
     - IEC 60287 conductor temperature, a transient thermal network, frictional heating and
       the thermal change of the clamping
   * - Limit states
     - armour yield and sheath strain envelopes, armour compression, coiling, sheaves,
       capacity checks of time series
   * - Fatigue
     - rainflow counting and Miner damage per wire and fibre for bulk, fretting, conductor,
       sheath and tape fatigue; long-term damage over a scatter of sea states;
       termination-adjusted fatigue
   * - Global to local
     - readers for CableDyn, RIFLEX and CSV time-history results; the ``cablexsec-batch`` command;
       hysteretic bending-law export for global analysis

Learning path
-------------

:doc:`installation` → :doc:`quickstart` → :doc:`user_guide` (the workflow through the Python
API) → :doc:`local_analysis_guide` (inputs, conventions and model choices in depth) →
:doc:`global_local_guide` (from a global analysis to fatigue). :doc:`examples` lists every
shipped script.

Validation
----------

Analytical gates, comparisons with independent methods, and comparisons with published
measurements (Ménard and Cartraud 2023, Witz 1996, Sævik 2011, Hu et al. 2022 and others) are
summarised in :doc:`validation`; the benchmark targets and results are in :doc:`benchmarks` and
the rules that govern them in :doc:`validation_policy`.

How to cite
-----------

If CableXSec contributes to published work, cite the software release as described in
:doc:`citing`.

.. toctree::
   :hidden:
   :caption: Getting started

   installation
   quickstart
   examples
   user_guide

.. toctree::
   :hidden:
   :caption: Guides

   local_analysis_guide
   global_local_guide
   project_guide
   uls_guide
   rig_test_guide
   radial_model
   template_66kv

.. toctree::
   :hidden:
   :caption: Reference

   cli
   api_python

.. toctree::
   :hidden:
   :caption: Validation

   validation
   benchmarks
   validation_policy

.. toctree::
   :hidden:
   :caption: Project

   changelog
   citing
   contributing
   development
