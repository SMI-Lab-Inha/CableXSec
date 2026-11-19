.. SPDX-License-Identifier: Apache-2.0

Quick start
===========

Load the bundled cable of Ménard and Cartraud (2023), compute its stiffness, a bending loop
and the armour fatigue damage of a curvature record:

.. code-block:: python

   import numpy as np
   import cablexsec as cx

   cable = cx.load_cable(cx.menard2023_yaml_path())
   stiffness = cx.compute_stiffness(cable)
   print(f"EA = {stiffness.ea_fixed_N / 1e6:.0f} MN, "
         f"EI full slip = {stiffness.ei_min_Nm2:.0f} N m^2, "
         f"EI no slip = {stiffness.ei_max_Nm2 / 1e3:.1f} kN m^2")

   curvature = 0.1 * np.sin(np.linspace(0.0, 6.0 * np.pi, 1801))  # three cycles, 1/m
   loop = cx.compute_mkappa(cable, curvature, tension_N=0.0, contact_model="bounded")
   print(f"peak moment {loop.moment_Nm.max():.0f} N m, "
         f"friction work {loop.W_diss_J_per_m:.1f} J/m over the history")

   kx = 0.02 * np.sin(np.linspace(0.0, 200.0 * np.pi, 4001))
   fatigue = cx.compute_fatigue(
       cable, kx, np.zeros_like(kx), tension_N=np.full_like(kx, 1.0e5)
   )
   for layer, damage in fatigue.D_total_per_wire.items():
       print(f"{layer}: largest Miner damage {damage.max():.3e} over the record")

The output is::

   EA = 280 MN, EI full slip = 3571 N m^2, EI no slip = 189.8 kN m^2
   peak moment 541 N m, friction work 218.0 J/m over the history
   inner_armour: largest Miner damage 1.774e-06 over the record
   outer_armour: largest Miner damage 3.005e-07 over the record

Every result carries a ``provenance`` record that names the method, the cited sources, the
verification gates and the limitations of the calculation; print ``fatigue.provenance`` to see
them. The :doc:`user_guide` walks through the whole workflow, and :doc:`examples` lists the
shipped scripts.
