.. SPDX-License-Identifier: Apache-2.0

Python API
==========

The public interface is the ``cablexsec`` namespace: ``import cablexsec as cx``. A calculation
loads a cable definition and calls a ``compute_*``, ``solve_*``, ``simulate_*`` or ``read_*``
entry point; every result object carries a ``provenance`` record. The tables below list the
entry points; their docstrings, shown here, give the parameters, units and limitations. The
:doc:`local_analysis_guide` explains how they fit together.

Cables
------

.. autofunction:: cablexsec.load_cable
.. autofunction:: cablexsec.menard2023_yaml_path
.. autofunction:: cablexsec.iea_task49_66kv_yaml_path

Section properties and contact state
------------------------------------

.. autofunction:: cablexsec.compute_stiffness
.. autofunction:: cablexsec.solve_section_state
.. autofunction:: cablexsec.compute_section_properties

Bending and stress
------------------

.. autofunction:: cablexsec.compute_mkappa
.. autofunction:: cablexsec.compute_stress
.. autofunction:: cablexsec.compute_component_response

Friction law
------------

The static/dynamic, rate- and contact-pressure-dependent armour friction
coefficient taken by ``compute_mkappa`` and ``compute_fatigue``
(``friction_law``, with ``time_s``), and the single spring-slider that shows
its closed forms.

.. autoclass:: cablexsec.FrictionLaw
   :members: rate_factor, pressure_factor_at, describe, is_neutral, needs_time
.. autofunction:: cablexsec.mechanics.friction_law.slider_history

Fatigue
-------

.. autofunction:: cablexsec.compute_fatigue
.. autofunction:: cablexsec.compute_long_term_fatigue
.. autofunction:: cablexsec.compute_conductor_fatigue
.. autofunction:: cablexsec.analysis.fatigue.curvature_path_out_of_plane_ratio

Thermal
-------

.. autofunction:: cablexsec.compute_thermal

Global results
--------------

.. autofunction:: cablexsec.read_series_csv
.. autofunction:: cablexsec.read_time_history_csv
.. autofunction:: cablexsec.read_riflex_ascii
.. autofunction:: cablexsec.read_cabledyn_sections
.. autofunction:: cablexsec.read_cabledyn_range_graph
.. autofunction:: cablexsec.read_cabledyn_lift
.. autofunction:: cablexsec.run_sea_state_batch
