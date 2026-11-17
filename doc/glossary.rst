.. SPDX-License-Identifier: Apache-2.0

Glossary
========

Terms, abbreviations and symbols as CableXSec uses them. Each entry gives the plain meaning
first and then the precise meaning in CableXSec, with the page or function where it is used.
All quantities are SI (m, N, Pa, s, 1/m, °C unless a name says otherwise); stress is
tensile-positive and pressure is positive in compression. The model behind the terms is set
out in :doc:`concepts`; inputs and conventions in depth are in :doc:`local_analysis_guide`.

Terms and abbreviations
-----------------------

.. glossary::

   along-wire model
      A finite-length armour model in which every helical wire is solved along its own helix
      through a specimen or near an anchor, so that slip can redistribute between stations of
      different curvature. Selected with ``along_wire=`` (:class:`~cablexsec.AlongWireSpec`)
      in :func:`~cablexsec.simulate_bending_test`, or ``solver="along_wire"`` in a
      :class:`~cablexsec.TerminationSpec`; see :doc:`local_analysis_guide`, Section 6.3. Its
      infinite-length limit is the distributed :term:`friction model`.

   analysis record
      One completed run stored in a :term:`project`: its inputs, result, engine snapshot,
      run identifier, time and :term:`fingerprint` (``AnalysisRecord``). Records are never
      edited or overwritten; see :doc:`project_guide`.

   anchored armour
      Armour wires held axially at an end fitting, a rig clamp or a clamped bend-stiffener
      root, so that friction builds up a mean wire force the sectional law cannot carry. The
      :term:`termination correction` and :func:`~cablexsec.simulate_rig_test` solve each
      anchored wire with a Coulomb bound per unit length and axial slip only. The termination
      correction and the :term:`along-wire model` can hold the wire end elastically instead
      (``k_anchor_N_m``, the stiffness per wire of an end-fitting potting; ``0`` is a free
      wire end); the rig-test anchor is rigid.

   annual damage
      The :term:`Miner sum` per year of service, weighted over the sea states of a
      :term:`scatter diagram`: ``Σ p_i s_year D_i / T_i`` with probability ``p_i``, record
      damage ``D_i`` and record duration ``T_i``, combined at each :term:`fixed material
      point` before the maximum is taken. Computed by
      :func:`~cablexsec.compute_long_term_fatigue` and :func:`~cablexsec.run_sea_state_batch`.

   annulus
      The space between the inner and outer polymer sheaths of a cable or flexible pipe, in
      which the armour lies. Its state, :term:`flooded annulus` or :term:`sealed annulus`, is
      declared in ``water_ingress.armour_annulus`` and decides which tension loads the armour
      and which :term:`S–N curve` applies (:doc:`local_analysis_guide`, Section 7.2).

   API 17J
      The American Petroleum Institute specification for unbonded flexible pipe. CableXSec
      quotes it for the 7.7 % allowable strain of polyethylene and the 0.67 tensile-armour
      utilisation in normal operation (:doc:`uls_guide`).

   Archard wear
      The wear law in which the worn volume is proportional to the normal load times the
      sliding distance, with a coefficient ``k``. CableXSec converts Archard coefficients to
      energy coefficients ``α = k/μ`` and computes wear as ``V = α Σ W`` over the dissipated
      energy of gross-slip cycles; see :term:`wear`.

   armour
      Layers of helically wound steel wires (round, flat or profiled) that carry tension and
      protect the cable. In a cable definition each armour layer is a ``helical`` layer with
      its wire count, wire size, :term:`helix radius`, :term:`lay angle` and :term:`lay
      direction`.

   Arrhenius coefficient
      The constant ``B = ΔW/k`` [K] that sets how fast a thermally activated ageing process
      speeds up with absolute temperature. It is the thermal parameter of the
      :term:`electro-thermal ageing` model of :func:`~cablexsec.compute_electrical_life`
      (``thermal_b_k`` in the insulation ``electrical`` block).

   axial stiffness
   EA
      The axial force per unit axial strain of the cable. :func:`~cablexsec.compute_stiffness`
      returns it with the ends held against twist (``ea_fixed_N``) and free to twist
      (``ea_free_N``); the difference is the tension–twist coupling of the helical layers
      (:term:`Witz–Tan stiffness matrix`).

   backbone
      The moment–curvature curve on first (virgin) loading from a straight, stress-free
      section. :func:`~cablexsec.export_bending_law` computes it at each tension with
      :func:`~cablexsec.compute_mkappa` and decomposes it into an :term:`Iwan model`; every
      later branch follows from it by the :term:`Masing rules`.

   bedding
      A polymer layer, extruded or taped, under or between armour layers. In a cable
      definition it is a ``cylindrical`` layer that transmits contact pressure and adds its
      own bending to the :term:`full-slip bending stiffness`.

   bellmouth
      The flared guide at a hang-off or riser end that limits the bending of the cable as it
      leaves the support. The near-bellmouth armour stresses measured on the Sævik (2011)
      riser are the test of the :term:`termination correction` (:doc:`benchmarks`, SV11).

   bend restrictor
      An articulated chain of rigid elements around the cable that locks at a set angle and so
      limits the curvature. :class:`~cablexsec.ArticulatedRestrictor` and
      :func:`~cablexsec.restrictor_curvature_field` give the non-proportional curvature field
      ``κ(x, t)`` near a restrictor (:doc:`local_analysis_guide`, Section 13.5).

   bend stiffener
      A tapered polyurethane sleeve at a hang-off that spreads the bending over a length of
      cable. :class:`~cablexsec.BendStiffener` with :func:`~cablexsec.stiffener_curvature_field`
      gives its curvature field from a large-deflection beam model; its root clamps the armour
      (see :term:`anchored armour`).

   bending direction
      The direction of the tensile extreme fibre of the section, ``θ = atan2(ky, kx)`` from the
      :term:`strain gradient`. The default :term:`friction model` ties the friction cap of every
      wire to it; :func:`~cablexsec.compute_neutral_axes` reports it at every sample.

   bending law
      A moment–curvature relation that a global analysis can carry.
      :func:`~cablexsec.export_bending_law` writes the hysteretic law of the section as an
      :term:`Iwan model` per tension (JSON, CSV and a table in kN and m). The law is planar,
      tension and annulus state are fixed within one law, and tension levels should be no more
      than about 20 kN apart (:doc:`global_local_guide`, Section 8).

   bending stiffness
   EI
      The bending moment per unit curvature [N m²]. For a cable with helical armour it depends
      on the slip state and lies between the :term:`full-slip bending stiffness` and the
      :term:`no-slip bending stiffness`; a :term:`secant stiffness` is one value for a chosen
      curvature range.

   bird-caging
   radial buckling
      Outward buckling of the armour wires under axial compression, restrained only by the
      hoop stiffness of the layers outside them. :func:`~cablexsec.compression_screen` gives
      the radial capacity ``P2`` and its interaction with hoop capacity and yield ``P2*``;
      :func:`~cablexsec.compute_coiling` checks it for a coiled cable (:doc:`uls_guide`).

   bit for bit
      Identical in every binary digit, not only to round-off. The documentation uses it where
      an option, at its neutral setting, must reproduce the result without it.

   blocked S–N curve
      A fatigue curve listed in the library but refused because its basis is not adequate
      (for example the lead-sheath curve). Naming it raises :class:`~cablexsec.BlockedSNCurveError`
      rather than returning zero damage (``sn_curves.BLOCKED_SN_CURVES``).

   bounded chain
   radial contact chain
      The default :term:`contact model` (``contact_model="bounded"``): a fixed-radius chain
      that passes the contact pressure from layer to layer, from the outside in. Each helical
      layer adds a :term:`capstan effect` pressure jump from its tension and its declared
      clamping jump ``P0``. It supports offset cores (:doc:`radial_model`, Section 1.1).

   bulk fatigue
      Fatigue of the wire material away from a contact, from the rainflow-counted stress at
      :term:`fixed material points` against the layer's ``bulk_sn`` curve. The bulk damage is
      ``D_bulk_per_wire`` and ``D_bulk_per_fibre`` of :func:`~cablexsec.compute_fatigue`.

   bundled cable
      A published cable construction shipped with the package as a YAML file, each with its
      source stated layer by layer, for example :func:`~cablexsec.menard2023_yaml_path` or
      :func:`~cablexsec.iea_task49_66kv_yaml_path` (:doc:`local_analysis_guide`, Section 4.1).

   cable definition
      A YAML file that describes a cable layer by layer, from the centre outwards, with a
      cited source for every property. :func:`~cablexsec.load_cable` reads it and checks it
      against the :term:`schema`; see :doc:`local_analysis_guide`, Section 4.1.

   CableDyn
      An open-source global line solver for mooring lines and dynamic cables. CableXSec reads
      its runs section by section with :func:`~cablexsec.read_cabledyn_sections`, its 0.4.0
      station tables with ``cablexsec.io.read_cabledyn_station_table``, and sums sea states
      with :term:`cablexsec-batch` (:doc:`global_local_guide`).

   cablexsec-batch
      The command-line program that computes the weighted annual armour damage of a set of
      CableDyn sea-state runs from a JSON specification. It writes a ``manifest.json`` with the
      :term:`SHA-256 digest` of every input, the annual damage at every fixed fibre and the
      section histories (:doc:`cli`; :doc:`global_local_guide`, Section 3).

   cantilever test
      See :term:`test arrangement`.

   capacity check
      A global time series placed sample by sample on the :term:`capacity envelope`:
      :func:`~cablexsec.check_capacity` returns the :term:`utilisation` history, its maximum
      and time, exceedances, the minimum-bend-radius check and, with ``blocks``, a
      :term:`Gumbel fit` of block maxima (:doc:`local_analysis_guide`, Section 8.1).

   capacity envelope
      The tension–curvature boundary within which every component stays within its limit.
      :func:`~cablexsec.compute_capacity` gives the first-yield and buckling envelope at a
      design factor of one; :func:`~cablexsec.build_capacity_envelope` gives the curvatures
      at a set of design utilisation levels per tension and names the governing component.

   capstan effect
      The inward pressure that a tensioned helical layer exerts on what lies inside it, as a
      rope does on a capstan. In the :term:`bounded chain` each layer adds the jump
      ``T tan²α/(2π r²)`` (Féret and Bournazel 1987, Eq. 5). The exponential change of the
      friction cap along the curved wire with the wire's own force, ``expm1(μ |sin α| ψ)`` for
      a rising and ``1 − exp(−μ |sin α| ψ)`` for a falling stress, is called the capstan term
      of the cap (``capstan=True`` in the along-wire model).

   carcass
      The interlocked metal strip on the inside of a flexible pipe that resists collapse; it
      is not fluid-tight. It can be declared as a ``profiled`` wire layer; carcass collapse is
      outside the scope of CableXSec (:doc:`user_guide`, Section 4).

   cathodic protection
      Corrosion protection of steel in seawater by a sacrificial anode or impressed current.
      Seawater S–N curves state whether they hold with it, for example
      ``saevik_2011_flat_steel`` (seawater, cathodic protection).

   characteristic utilisation
      A utilisation at a design factor of one, the ratio ``κ/κ_max(T)`` on the envelope of
      :func:`~cablexsec.compute_capacity`. :func:`~cablexsec.compute_sheave` reports it beside
      the design utilisation on the capacity envelope.

   CI
   continuous integration
      Automatic running of the test suite on every change to the repository; the "Python CI"
      badge of the README shows its status.

   CIGRE TB
      A Technical Brochure of CIGRE, the international council on large electric systems. TB
      610 defines the wet and dry cable designs; TB 862 gives mechanical tests; TB 880 is the source of a bundled 220 kV export cable
      (:func:`~cablexsec.cigre_tb880_yaml_path`).

   clamp boundary layer
      The length from an anchor over which the friction-fed wire force decays to the sectional
      value, ``ℓ_x = ½ √(L_s P)`` with :term:`slip length` ``L_s`` and pitch ``P``.
      :func:`~cablexsec.simulate_rig_test` reports it per layer and places diagnostic gauges
      inside it (:doc:`rig_test_guide`, Section 3).

   clamping pressure
      See :term:`contact pressure` and :term:`manufacturing clamping`.

   coiling
      Storing a cable in a static tank, which bends and twists it together: a tank of
      diameter ``D`` imposes the curvature ``2/D`` and the twist ``2/D`` (one twist per turn).
      :func:`~cablexsec.compute_coiling` gives the armour stresses and the bird-caging and
      first-yield checks; the absolute stresses depend on the core lay, which must be stated
      (:doc:`benchmarks`, coiling).

   component
      Any layer of the cable cross-section: conductor, insulation, screens, sheaths, tapes and
      each armour layer. :func:`~cablexsec.compute_component_response` gives the stress,
      strain and utilisation of every component against the limits its material gives.

   compression screen
      The armour-in-compression check of :func:`~cablexsec.compression_screen`: lateral
      buckling (Sævik and Thorsen 2017; Li and Vaz 2021), radial buckling, hoop capacity and
      wire yield, with a friction gate and a :term:`design utilisation factor`.
      Compression needs a negative driving tension: effective for a flooded annulus, true wall
      for a sealed one (:doc:`uls_guide`).

   conductor
      The copper or aluminium core of a power core that carries the current, solid or
      stranded. In a cable definition it is a ``cylindrical_offset`` layer, optionally with
      resolved ``strand_layers``; its fatigue is computed by
      :func:`~cablexsec.compute_conductor_fatigue`.

   conductor fatigue
      Fatigue of the wires of a :term:`stranded conductor`, from the stick–slip strain of each
      wire ring (Karlsen-Husøy et al. 2026 operator) counted at eight fixed azimuths, with an
      optional :term:`trellis contact` term. :func:`~cablexsec.compute_conductor_fatigue`;
      :doc:`local_analysis_guide`, Section 10.

   contact model
      The calculation that turns tension, pressure, clamping and temperature into the contact
      pressures of the layers: ``"bounded"`` (the :term:`bounded chain`, default of the bending,
      stress and fatigue functions), ``"qp"`` (coupled concentric equilibrium with Poisson
      effects and contact opening, default of :func:`~cablexsec.solve_section_state`) and
      ``"solved"`` (the :term:`solved radial contact state`). The models can differ by tens of
      per cent on the same cable (:doc:`concepts`).

   contact pressure
      The normal pressure between an armour layer and its neighbours, which sets the friction.
      CableXSec works with the friction-setting pressure ``P1 + P2``, the sum of the pressures
      on the inner and outer faces of the layer referred to the helix radius. Its sources are
      :term:`manufacturing clamping`, tension, external pressure and temperature
      (:doc:`concepts`).

   contra-helical layers
      Two armour layers laid in opposite directions so that their torques under tension
      largely cancel. Where they touch directly, their wires cross at the sum of their lay
      angles (see :term:`crossing`).

   core lay
      The helical laying of the power cores of a three-core cable about the cable axis, with
      core lay angle ``β`` and pitch, declared in the ``core_lay`` block. It is required for
      laid-core friction (``core_friction``) and changes the torque and the strains of layers
      inside the cores.

   core slip state
      How a laid power core moves in the section, which sets the strain of the screens, sheaths
      and tapes inside it: ``"stuck"`` (bonded to the section, plane-section upper bound),
      ``"free"`` (slides along its helix, full-slip lower bound) or ``"frictional"`` (the
      laid-core friction law). A required input of :func:`~cablexsec.compute_core_layer_strain`
      and related functions (:doc:`local_analysis_guide`, Section 13.1).

   corrugated sheath
      A metallic sheath (usually aluminium) with circumferential corrugations, which acts as a
      radial water barrier. It is a ``cylindrical_shell`` layer with a corrugation factor
      ``f_corrugation`` and an :term:`SCF`; their product is the strain transfer at the
      corrugation root. :func:`~cablexsec.compute_shell_stress`,
      :func:`~cablexsec.compute_shell_fatigue`.

   Coulomb friction
      Dry friction in which the tangential force cannot exceed the friction coefficient times
      the normal force; below that cap the surfaces stick. In CableXSec each armour wire is a
      :term:`spring–slider` whose friction stress saturates at its cap once it slips (Coulomb
      saturation).

   critical distance
      The theory of critical distances: fatigue at a notch or contact is assessed with the
      stress at a material length below the surface rather than the peak stress. The default
      trellis rule of :func:`~cablexsec.compute_conductor_fatigue` uses the point method at
      depth ``L/2``, ``L = (1/π)(ΔK_th/Δσ₀)²`` (:class:`~cablexsec.CriticalDistanceSpec`).

   critical plane
      The plane through a material point on which a multiaxial fatigue parameter is largest
      over a load cycle. :func:`~cablexsec.compute_swt_cycle` finds the sampled critical plane
      of the :term:`SWT parameter` from coincident stress and strain histories
      (:doc:`tutorials/04_contact_and_fretting`, Section 8).

   critical wire and fibre
      The wire (by azimuth ``phi_critical_deg``) and the fibre on it (``fibre_critical_deg``)
      with the largest fatigue damage, reported with ``critical_layer`` by
      :func:`~cablexsec.compute_fatigue` and :func:`~cablexsec.compute_long_term_fatigue`.

   crossing
      A point where wires of two touching, opposite-laid layers cross; the contact there is a
      :term:`Hertz contact`. The crossing angle is the sum of the two lay angles and the
      crossing force is the interface line load divided by the crossings per unit length
      (:doc:`local_analysis_guide`, Section 5, Step 4).

   curvature
      The rate of change of direction of the cable axis [1/m], the inverse of the bend
      radius. For fatigue it must be given as two signed components, never as a magnitude; see
      :term:`strain gradient`, :term:`rotation curvature` and :term:`signed curvature
      components`.

   cylindrical_offset layer
      The layer type of a component of the power cores (conductor, insulation, screen, core
      sheath), declared once for ``n_cores`` identical cores at ``r_offset_m`` from the cable
      axis. The other layer types are ``helical``, ``helical_stranded``, ``cylindrical`` and
      ``cylindrical_shell`` (:doc:`local_analysis_guide`, Section 4.1).

   damage-equivalent stress range
   DEL
      The constant stress range that gives the same Miner damage as a counted spectrum in a
      chosen number of cycles, ``Δσ_eq = (Σ n_i Δσ_iᵐ / N_eq)^(1/m)`` for a single-slope S–N
      curve of slope ``m``; DEL stands for damage-equivalent load. ``fibre_fatigue`` returns the
      1 Hz value (one cycle per second) per fibre (:doc:`tutorials/03_global_to_fatigue`,
      Section 6).

   design fatigue factor
   DFF
      A factor of at least one that divides the allowable damage: the design life is
      ``1/(DFF · D_year)``. Passed as ``design_fatigue_factor`` to
      :func:`~cablexsec.compute_long_term_fatigue`; the IEA 66 kV template quotes DFF 10.

   design load case
   DLC
      A combination of environmental and operating conditions defined by a wind-turbine design
      standard (IEC 61400-3) for which loads are computed. DLC numbers appear in the list of
      planned benchmarks (:doc:`benchmarks`).

   design utilisation factor
      The fraction ``η`` of a capacity allowed in design. The armour factors of DNV-RP-F401
      are ``cx.ARMOUR_UTILISATION_FACTORS`` (0.67 normal operation, 0.78 installation, 1.00
      abnormal); other components take ``cx.COMPONENT_UTILISATION_FACTORS`` (0.80, 1.00, 1.00).
      The design utilisation is the utilisation divided by ``η`` (:doc:`uls_guide`, Section 4).

   direct tensile strain
      The axial strain of the cable without Poisson deformation (``axial_strain_kind="direct"``).
      When present in a :term:`load case series` it drives the wire stress instead of tension;
      a total mean axial strain is rejected (:doc:`local_analysis_guide`, Section 4.3).

   DNV-RP-C203
      The DNV recommended practice for fatigue design of offshore steel structures. CableXSec
      uses its T curve (``dnv_rpc203_t_curve``), its thickness correction and its mean-stress
      range factors (``"dnv_base"``, ``"dnv_pwht"``).

   DNV-RP-F401
      The DNV recommended practice for electrical power cables in subsea applications. Its
      armour utilisation factors are the default :term:`design utilisation factor` of the
      compression screen and the capacity envelope.

   effective tension
      The axial force of a submerged line that includes the hydrostatic pressure on its ends,
      the force a global line model reports: ``T_e = T_w + p_o A_o − p_i A_i``. Fatigue functions
      refuse a history marked as effective until it is converted with
      ``series.with_true_tension(...)``, except that for a :term:`flooded annulus` the
      effective tension is the force that loads the armour (:doc:`concepts`).

   electro-thermal ageing
      Degradation of polymer insulation under electric field and temperature together.
      :func:`~cablexsec.compute_electrical_life` returns the relative life ``L/L_0`` of XLPE
      insulation from the Mazzanti (2007) model with an :term:`Arrhenius coefficient`, a
      voltage-endurance coefficient and their synergism; it is a screening factor, not a
      calibrated life.

   end fitting
      The termination at a cable end that anchors the armour and seals the cable. Its
      effect enters through the :term:`termination SCF` and, for anchored armour, the
      :term:`termination correction`.

   energised
      Carrying current, so that the cable is heated. :class:`~cablexsec.ThermalHistory` couples
      the layer temperatures of an energised cable into clamping, stiffness and fatigue; the
      unenergised state, all layers at ``T_ref``, reproduces the ordinary results bit for bit
      (:doc:`local_analysis_guide`, Section 7.3).

   engine snapshot
      The calculation environment stored with every :term:`analysis record`: CableXSec version,
      source digest, Python and platform, dependency versions and the registered material and
      curve data. A change to any of them makes the record :term:`stale <stale record>`.

   exchange CSV
      CableXSec's own CSV format for a :term:`load case series`, with SI units and
      strain-gradient columns, written by :func:`~cablexsec.write_series_csv` and read by
      :func:`~cablexsec.read_series_csv`; a round trip preserves the tension and strain kinds.

   ExtrapolationWarning
      The warning issued when an input lies outside the range a model was validated against,
      for example a curvature above wire yield in :func:`~cablexsec.compute_mkappa`. The result
      is still returned; the message names the bound and its source.

   extreme fibre
      The point of the section farthest from the :term:`neutral axis` on the tensile or the
      compressive side, where the bending strain is largest. The friction cap of a wire is
      largest there, so the extreme-fibre wire is the last to slip.

   fatigue
      Damage of a material under repeated stress cycles, below its static strength. CableXSec
      counts it with :term:`rainflow counting` and the :term:`Miner sum` at :term:`fixed
      material points` for armour, conductor, screens, sheaths and tapes.

   FBG
   fibre Bragg grating
      An optical strain sensor written into a fibre. The FBG armour-stress measurements of the
      Sævik (2011) riser are the only measured armour stresses in the benchmark register
      (:doc:`benchmarks`, SV11).

   FE
   finite element
      A numerical method that divides a body into elements. CableXSec is semi-analytical and
      does not use FE contact; published FE results enter the benchmark register as
      :term:`model-to-model comparisons <model-to-model comparison>`. The plane trefoil model of
      the :term:`solved radial contact state` is a small FE model of the three-core bundle.

   fibre
      A material point on the perimeter of a wire, at a fixed angle measured from the outward
      radial direction (0° outward, 180° inward). :func:`~cablexsec.compute_fatigue` counts 32
      fibres per wire by default (``n_fibres``); see :term:`fixed material point`.

   fill ratio
      The share of the circumference of a helical layer occupied by its wires,
      ``n d / (2π r cos α)``; adjacent wires touch at a fill of one. A layer that overfills is
      rejected, and :func:`~cablexsec.design_torque_balance` keeps the fill within bounds.

   filler
      Material that fills the spaces between the laid cores of a three-core cable. It is
      declared as an interstitial ``cylindrical`` layer; the solved radial state meshes it as
      solid or leaves it void (``filler=``), and the three-core contact state depends strongly
      on that choice (:doc:`radial_model`, Section 3.4).

   fingerprint
      A SHA-256 digest over the inputs of a run (cable definition, request and history) and its
      engine snapshot, stored as ``record.fingerprint`` and used to detect a :term:`stale
      record`.

   first yield
      The load at which the most stressed point of a component first reaches its yield or
      0.2 % :term:`proof stress`. Wire first yield, with the same slip-front stress as bending
      and fatigue, is one mechanism of :func:`~cablexsec.compute_capacity`.

   fixed material point
   fixed material points
      A point of the material that keeps its position, every wire at its own azimuth and every
      :term:`fibre` on it. CableXSec counts fatigue at each such point separately, because the
      largest stress round the circumference can stay constant while every material point
      cycles when the bending direction rotates (:doc:`concepts`).

   flexible pipe
      An unbonded pipe of polymer sheaths and helical metal armour used for risers and
      flowlines. CableXSec analyses its tensile armour; two bundled cases are the Witz (1996)
      2.5-inch pipe and the Sævik (2011) riser.

   flooded annulus
      An armour annulus open to seawater (a :term:`wet design`). Hydrostatic pressure acts on
      every face of the layers, so it adds almost no clamping (``p_ext_Pa=0.0``), the
      :term:`effective tension` loads the armour, and a seawater S–N curve applies
      (:doc:`local_analysis_guide`, Section 7.2).

   FLS
   fatigue limit state
      The design check against fatigue failure over the service life. The CableXSec fatigue
      functions and :func:`~cablexsec.compute_long_term_fatigue` supply its damage; see also
      :term:`ULS`.

   four-point bending
      See :term:`test arrangement`.

   FOWT
   floating offshore wind turbine
      A wind turbine on a floating platform. The dynamic power cables of floating wind farms
      are the main application of CableXSec.

   fretting
      Damage at a contact under small cyclic slip: wear, crack initiation, or both. CableXSec
      assesses it at armour crossings and wire-on-polymer faces with
      :func:`~cablexsec.compute_fretting`, and at conductor crossings with the
      :term:`trellis contact` rule (:doc:`local_analysis_guide`, Section 11).

   fretting knock-down
      The reduction of fatigue life at a contact in partial slip. CableXSec multiplies the
      stress range of partial-slip cycles by the Nishioka–Hirakawa notch factor
      ``K_f = 1 + 2μ p_m (1 − e^(−δ/k)) / σ_a,ref`` with an assumed slip constant
      ``k = 2.7 µm``; gross slip has no knock-down because wear removes initiated cracks.

   fretting regime
      The class of a slip cycle at a contact: :term:`stick`, :term:`partial slip` (amplitude
      up to the :term:`transition amplitude`) or :term:`gross slip` (above it).
      :func:`~cablexsec.compute_fretting` classifies every rainflow-counted slip cycle and
      reports the share of each regime per face (``FrettingFace``).

   friction cap
      The largest friction stress a slipping armour wire can carry (Papailiou 1997):
      proportional to the contact pressure and to ``expm1(μ |sin α| ψ)`` for a rising stress
      and ``1 − exp(−μ |sin α| ψ)`` for a falling one (signed capstan term), with
      ``ψ = asin c`` and ``c = |cos(φ − φ_b)|`` for a wire at azimuth ``φ`` and bending
      direction ``φ_b``. It is zero at the neutral axis and largest at the extreme fibre; also
      called the Papailiou cap.

   friction coefficient
      The ratio ``μ`` of tangential to normal force at sliding. Library values are properties of
      the wire material against its neighbour (for example ``mu_on_hdpe`` 0.20); ``mu=`` overrides
      them. A measured bending loop fixes only the product of ``μ`` and the clamping pressure
      (:doc:`tutorials/02_bending_hysteresis`).

   friction model
      The armour stick–slip law of :func:`~cablexsec.compute_fatigue` and
      :func:`~cablexsec.compute_mkappa` (``friction_model``): ``"direction_tied"`` (default; one
      spring–slider per wire with the cap of the current bending direction), ``"distributed"``
      (friction solved along one pitch with an :term:`interface shear stiffness`) or
      ``"progressive"`` (each wire as an :term:`Iwan model` population that leaves stick
      gradually); see :doc:`local_analysis_guide`, Section 9.

   friction stress
      The axial wire stress transferred by friction from bending: the stick stress
      ``k r κ c`` below slip and the :term:`friction cap` in slip. With the axial stress from
      tension and the local bending of the wire it makes up the wire stress of the fatigue
      route.

   friction work
      The energy dissipated by friction per metre of cable, the area of a closed
      moment–curvature loop. ``result.work`` of :func:`~cablexsec.compute_mkappa` holds the
      cumulative stored energy and friction work; ``W_diss_J_per_m`` is the friction work over
      the whole supplied history, conditioning cycles included.

   frictional heating
      Heat released by armour and core friction in cyclic bending.
      ``cablexsec.analysis.frictional_heating.simulate_frictional_heating`` feeds it into the
      :term:`thermal network`, one-way or two-way coupled with the clamping.

   full slip
      The state in which every wire of a layer slips; the moment then rises with the
      :term:`full-slip bending stiffness`. It is reached at the :term:`full-slip curvature`.

   full-slip bending stiffness
      The lower bound of the bending stiffness, with every wire, core and layer bending about
      its own axis (Witz and Tan 1992): ``ei_min_Nm2`` of :func:`~cablexsec.compute_stiffness`.
      It barely depends on friction or pressure, which makes it a calibration-free check
      against a test. A global model that uses it gives upper estimates of local damage.

   full-slip curvature
      The curvature at which the last wire, on the tension-side extreme fibre, starts to slip,
      ``κ_on expm1(μ |sin α| π/2)/(μ |sin α|)`` for a layer with slip onset ``κ_on``. Returned by
      ``full_slip_curvature`` of ``cablexsec.mechanics.papailiou`` and by
      ``armour_slip_curvatures``.

   gap
      A labelled development item of the benchmark register and the guides, for example gap
      G1 (rectangular tendons), T1 (test arrangement), T2 (along-wire redistribution) or P5
      (restrictor fields). The label ties a capability to its checks in :doc:`benchmarks`.

   gate
   verification gate
      An identifier of a check in the validation scorecard (for example ``A11``, ``BM1``,
      ``SV11``). Every result's :term:`provenance` lists the gates that back it; the gates,
      their targets and status are in :doc:`benchmarks` and :doc:`validation_policy`.

   geodesic
      The shortest path on a surface. A wire that slips freely on a bent cylinder moves towards
      the geodesic, with transverse slip; ``transverse_map="geodesic"`` scales the transverse
      part between the :term:`loxodrome` and the frictionless geodesic slip (Out and von Morgen
      1997) (:doc:`local_analysis_guide`, Section 14).

   geometry override
      A separate YAML file that supplies geometry withheld (``null``) in a cable definition,
      merged by layer name with ``load_cable(path, geometry_override=...)``.

   global line model
   global analysis
      A model of the whole cable as a slender rod under waves, current and vessel motion. It
      needs section properties (``EA``, ``EI``, ``GJ``) and returns, at every point, a history
      of tension, two curvature components and possibly pressure and temperature, which
      CableXSec reads (:doc:`concepts`).

   global-to-local
      The workflow from a global analysis to local results: read the history at a section,
      compute wire stresses at fixed material points and sum the damage. Stage 1 is one way
      (CableXSec reads CableDyn); stage 2, the two-way hysteretic element, is open
      (:doc:`global_local_guide`).

   gross slip
      A fretting regime in which the whole contact slides in each cycle; the slip amplitude
      exceeds the :term:`transition amplitude`. Its energy drives wear; it carries no
      :term:`fretting knock-down`.

   Gumbel fit
      A fit of the extreme-value (Gumbel) distribution to maxima, by the method of moments.
      :func:`~cablexsec.capacity_extremes` and :func:`~cablexsec.gumbel_fit` give the most
      probable maximum, the mean and a fractile of the utilisation over seeds or blocks.

   hang-off
      The upper end of a dynamic cable at the floating platform, usually with a bend stiffener.
      Node 1 (End A) of a CableDyn line; it carries the top tension and does not normally go
      into compression.

   HDPE
   MDPE
   PE
      High-density, medium-density polyethylene and polyethylene in general, the usual sheath
      materials. Library materials such as ``hdpe`` carry their source; ``hdpe`` has a
      registered modulus–temperature law.

   helix radius
      The radius of the centre line of a helical wire about the cable axis, ``r_helix_m`` in a
      cable definition (``r`` or ``R`` in the guides).

   Hertz contact
      The elastic contact of two curved bodies, with an elliptical footprint and a peak
      pressure ``p0``. :func:`~cablexsec.mechanics.hertz_contact.solve_contact` solves armour
      crossings; :func:`~cablexsec.compute_hertz_axis_stress` and
      :func:`~cablexsec.compute_hertz_subsurface_stress` give the subsurface stress. At
      GPa-level pressures the elastic values are indicators: real crossings yield and flatten.

   hysteresis loop
   moment–curvature loop
      The closed path of moment against curvature under cyclic bending; its area is the
      friction work dissipated per cycle. :func:`~cablexsec.compute_mkappa` returns it for any
      curvature history (:doc:`tutorials/02_bending_hysteresis`).

   identified
      Labels a parameter fitted to a measured response, for example the 1.037 MPa initial
      pressure of the Witz pipe identified on its 0-bar loop. Under :doc:`validation_policy`,
      Rule 3, a response used for identification cannot validate the model.

   IEA Wind Task 49 66 kV template
      A public three-core wet-design 66 kV cable built from Janocha et al. (2024) and IEA Wind
      Task 49 (:func:`~cablexsec.iea_task49_66kv_yaml_path`). Its source publishes no
      manufacturing clamping, so it declares zero and warns (:doc:`template_66kv`).

   IEC 60287
      The IEC standard for current rating of electric cables. :func:`~cablexsec.compute_thermal`
      evaluates its steady thermal ladder with the external resistance ``T4``; the
      :term:`thermal network` gives the transient response through the same ladder.

   initial state
      The ``initial_state`` block of a cable definition: a uniform external pressure on the
      assembled cable before loading, with an ``identification`` class (measured,
      manufacturing specification, published calibration, identified) and a source. The Ménard
      cable carries 0.3 MPa, a published calibration.

   insulation
      The dielectric layer round the conductor, usually :term:`XLPE`, between the conductor
      and insulation screens. A ``cylindrical_offset`` layer; its electrical block feeds
      :func:`~cablexsec.compute_electrical_life`.

   interface shear stiffness
      The shear stiffness ``k̂ = ξ/(1 − ξ)`` (normalised) of the elastic interface between a
      wire and the layer it bears on, used by the distributed :term:`friction model` and by
      ``stick="spring"`` of the along-wire solver. It follows from the :term:`shear
      coefficient` ξ.

   Iwan model
      A set of parallel elastic–perfectly plastic elements (spring–sliders) that represents a
      hysteretic law (Iwan 1966). The exported :term:`bending law` is an Iwan model plus the
      full-slip stiffness; it follows the :term:`Masing rules` exactly. The progressive friction
      model gives each wire an Iwan population.

   JONSWAP
      A standard spectrum of irregular sea waves defined by significant wave height, peak
      period and a peak-enhancement factor. :func:`~cablexsec.jonswap_series` shapes a synthetic
      curvature and tension record with it, for demonstrations and tests only.

   laid three-core cable
      A cable with three power cores laid helically about its axis, with fillers between them
      and armour outside. The cores are one ``cylindrical_offset`` definition; their lay is the
      :term:`core lay`.

   lateral buckling
      Sideways buckling of armour wires in axial compression after they slip at the neutral
      axis. The capacity of :func:`~cablexsec.compression_screen` is the smaller of the
      Sævik–Thorsen lower bound ``P1`` and the knocked-down Li–Vaz limit (:doc:`uls_guide`).

   lateral relief
      The reduction of the axial friction bound of a wire that also carries a transverse line
      load (vector Coulomb). :class:`~cablexsec.LateralRelief` applies it as a sensitivity; on
      the Sævik riser it changes the ranges by 1–3 % (:doc:`local_analysis_guide`, Section 6.4).

   lay angle
      The angle between a helical wire (or core or tape) and the cable axis, ``α``
      (``lay_angle_deg``); ``β`` for the core lay and the tape angle. The friction cap, the
      stick stiffness and the torque all depend on it.

   lay direction
      The hand of a helix, ``right_hand`` or ``left_hand`` (``lay_direction``), with sign
      ``s = +1`` or ``−1``. It sets the sign of the layer torque and the crossing pattern of
      contra-helical layers.

   lay length
   pitch
      The axial length over which a helix makes one full turn, ``P = 2π r / tan α``. The section
      law assumes the curvature is uniform over about one lay length.

   lazy wave
      A dynamic cable configuration with a buoyant middle section that forms an arch between
      the hang-off and the seabed, with a :term:`sag bend` below and a hog bend over the arch.

   lead-in
      The initial part of a history used to build up the friction state of the wires before
      counting. ``count_from_sample=`` of :func:`~cablexsec.compute_fatigue` runs the whole record
      and counts from that sample on (:doc:`local_analysis_guide`, Section 5, Step 3).

   line contact
      A contact along a line, such as a wire bearing on a flat polymer layer. In
      :func:`~cablexsec.compute_fretting` a wire on polymer is a Hertz line contact with a regime
      boundary set by the support compliance of the section law.

   load case series
      The history of a global analysis at one section: ``LoadCaseSeries`` with uniform time
      steps, two curvature components, tension, optional direct strain and temperature, and its
      :term:`tension kind`. Every reader returns one (:doc:`local_analysis_guide`, Section 4.3).

   load sharing
      The division of the cable tension among its layers. The fixed-radius route shares it by
      equal axial strain (EA-weighted, twist fixed); on the solved route of
      :func:`~cablexsec.compute_section_properties` every layer's force is recovered from its own
      strain. :func:`~cablexsec.compute_load_sharing` reports the shares.

   long-term fatigue
      The fatigue damage over the design life from a :term:`scatter diagram` of sea states,
      each a :class:`~cablexsec.FatigueLoadCase` with its probability.
      :func:`~cablexsec.compute_long_term_fatigue` combines the damage at each fixed fibre before
      taking the maximum (:doc:`local_analysis_guide`, Section 5, Step 5).

   loose layer
      A layer with no axial stiffness in the fixed-radius model, such as a copper wire or tape
      screen (``structural: false``). It still transmits pressure; its stress is its modulus
      times the strain it follows, and it is never the critical component.

   loxodrome
      A helix that crosses every generator of a cylinder at the same angle. The default fatigue
      path keeps the wires on the loxodrome of the bent cable (``transverse_ratio=0``), with no
      transverse slip; see :term:`geodesic` and :term:`transverse slip`.

   manufacturing clamping
      The contact pressure the armour carries at zero applied tension, left by winding,
      extrusion and sheathing. It is an input, not a prediction: each armour layer declares
      ``armour_winding_stress_Pa`` or ``initial_contact_pressure_Pa`` (the jump ``P0``) with a
      basis (measured, specification, published calibration, assumed, representative or not
      published) and a source. :func:`~cablexsec.clamping_state` resolves it
      (:doc:`local_analysis_guide`, Section 4.4).

   Masing rules
      Rules that build every unloading and reloading branch of a hysteretic law from its
      :term:`backbone`: a branch from a reversal is the backbone scaled by two about the reversal
      point, and a branch that meets an earlier one continues it (memory). The Iwan bending law
      obeys them; the strain-life route uses Masing branches on the cyclic stress–strain curve.

   material registry
      The library of material constants with a citation per value
      (``cablexsec.materials.library``). :func:`~cablexsec.register_material` adds a user
      material for the session; S–N curves and strain-life constants have registries of their
      own.

   MBL
   minimum breaking load
      The rated tension at which a cable breaks. CableXSec shows it beside the
      tension capacity as reference metadata (``aggregate_properties``) and never uses it as a
      limit.

   MBR
   minimum bend radius
      The smallest bend radius a cable may take, stated separately for dynamic service and
      installation (``MBR_dynamic_m``, ``MBR_installation_m``). :func:`~cablexsec.check_capacity`
      reports the check ``κ · MBR`` and :func:`~cablexsec.compute_sheave` the sheave radius over
      the installation MBR.

   mean-stress correction
      A conversion of each fatigue cycle to the test condition of an S–N curve, accounting for
      its mean stress. :class:`~cablexsec.MeanStressSpec` offers Goodman, Gerber, Soderberg
      (with the tensile or yield strength), SWT, Walker and the DNV-RP-C203 factors; by default
      no correction is applied to armour (:doc:`local_analysis_guide`, Section 4.2).

   metallic sheath
      A continuous metal tube (lead or aluminium) that keeps water out of the cores, declared as
      a ``cylindrical_shell``. A dry design must contain one; lead-sheath fatigue is blocked.

   Mindlin partial slip
   Cattaneo–Mindlin
      The tangential loading of an elastic contact below gross sliding: an annulus at the edge
      slips while the centre sticks. ``MindlinContact`` follows it at armour crossings with the
      tangential stiffness ``K_t`` of the Hertz contact; the energy per cycle gives the partial-
      slip wear and regime (:doc:`local_analysis_guide`, Section 11).

   Miner sum
      The linear damage sum ``D = Σ n_i / N_i`` of counted cycles ``n_i`` against the cycles to
      failure ``N_i`` of the S–N curve at their ranges; failure is predicted at ``D = 1``. Half
      cycles count 0.5.

   MissingClampingStateWarning
      The warning issued when a friction-dependent analysis forms the armour contact pressure of
      a cable with no declared :term:`manufacturing clamping`, or with zero clamping because none
      is published. Its friction results then rest on tension-induced contact alone and are
      sensitivities, not cable properties.

   model-to-model comparison
      A comparison with another model (published FE or a published analytical model) at matched
      inputs. It can expose errors but is not validation; the register keeps
      it separate from measured comparisons (:doc:`validation_policy`).

   named error
      An exception of a CableXSec class, raised instead of returning a partial or approximate
      result: ``CableInputError``, ``CableSchemaError``, ``CableLookupError``, ``CableSolverError``,
      ``PhysicalRangeError`` and ``BlockedSNCurveError``, all subclasses of
      :class:`~cablexsec.CableXSecError`. The message names the file, column, layer or value.

   neutral axis
      The line across the section with zero bending strain. The friction cap of a wire is zero
      there, so slip starts on the neutral axis; under biaxial bending and slip the strain and
      stress neutral axes differ, and :func:`~cablexsec.compute_neutral_axes` reports both.

   no-slip bending stiffness
      The upper bound of the bending stiffness, with the wires sticking and following the
      bending of the section, reduced by the :term:`shear coefficient` ξ of each armour layer:
      ``ei_max_Nm2`` of :func:`~cablexsec.compute_stiffness`. The loop starts with this slope.

   numba kernels
      Compiled routines for the per-wire friction-stress recursion and the rainflow count,
      installed with the ``fast`` extra (``pip install "cablexsec[fast]"``). Without numba, or
      with ``CABLEXSEC_DISABLE_NUMBA=1``, the :term:`reference implementation` runs with
      identical results, more slowly (:doc:`local_analysis_guide`, Section 9).

   OD
   outer diameter
      The overall diameter of a layer or of the cable. The 66 kV template lists layer ODs; the
      mass properties report the cable OD and the restrictor field takes ``cable_od_m``.

   partial slip
      For a contact, a fretting regime in which part of the contact sticks and the slip
      amplitude is at most the :term:`transition amplitude`; these cycles carry the
      :term:`fretting knock-down`. For an armour layer, the bending state between slip onset
      and full slip, in which only the wires near the neutral axis slip.

   plane sections
      The beam assumption that cross-sections stay plane, so that the axial strain is linear
      over the section. ``ei_plane_section_Nm2`` is the stick bending stiffness with that
      assumption and ξ = 1; the ``"stuck"`` core slip state is its upper bound for core layers.

   polymer relaxation factor
      A factor ``f`` (0 < f ≤ 1) on the polymer moduli of every radial solve, a relaxed-modulus
      proxy for hot polyethylene and XLPE under sustained clamping and thermal stresses
      (``polymer_relaxation_factor``). The stick stiffness of fast bending keeps the instantaneous
      modulus.

   power core
      One insulated conductor with its screens and core sheath. A three-core cable lays three
      power cores about its axis; see :term:`laid three-core cable`.

   PP
   PPY
      Polypropylene, and polypropylene yarn, used for fillers and beddings (``pp_filler``).

   preflight
      A check of every analysis of a project against its cable and load case without running
      it: ``cablexsec.preflight.preflight(project)`` returns errors (combinations the package
      refuses) and warnings (runs that rest on an assumption) (:doc:`project_guide`).

   progressive friction model
      See :term:`friction model`.

   project
   project file
      A :class:`~cablexsec.Project` keeps cable models, load cases, analysis requests and
      completed runs together, saved as one self-contained UTF-8 JSON file (named ``.cxproj`` in
      this manual) with a SHA-256 checksum (:doc:`project_guide`).

   proof stress
      The 0.2 % proof stress: the stress at which a metal without a sharp yield point has a
      permanent strain of 0.2 %. The conductor and metallic screens are checked against it
      (``sigma_y_Pa``).

   provenance
      The record attached to every result object: the method, its citations, the verification
      :term:`gates <gate>` that back it and its active limitations
      (:class:`~cablexsec.Provenance`). Print ``result.provenance`` to read it.

   PU
      Polyurethane, the material of bend stiffeners; the stiffener model takes its modulus as
      linear.

   QP
      Quadratic programme. The ``"qp"`` :term:`contact model` solves the concentric section with
      frictionless unilateral contact as a quadratic programme; the optional ``contact`` extra
      installs the Clarabel solver.

   rainflow counting
      An algorithm that extracts closed stress cycles and residual half cycles from an
      irregular history (ASTM E1049). CableXSec counts every fibre separately with a compiled
      counter that reproduces the reference counter cycle for cycle.

   RAO
      Response amplitude operator: the motion of a vessel per unit wave amplitude as a function
      of frequency. The genuine CableDyn test runs move the hang-off with illustrative RAOs.

   rectangular tendon
   profiled wire
      Armour of flat strips (``wire_shape: rectangular``) or of interlocked profiles
      (``profiled``, for example a carcass), declared with width, thickness and, for profiles,
      the source's area and inertias. ``cx.WireSection`` carries the section; round wires are
      unchanged bit for bit (:doc:`local_analysis_guide`, Section 4.1).

   reference implementation
      The pure-Python routines that the compiled :term:`numba kernels` must reproduce; tests
      compare the two cycle for cycle.

   representative
      A clamping basis or cable label for a value chosen to be typical, not taken from a
      particular cable, for example the 2.0 MPa ``P0`` of the 33 kV templates.

   RIFLEX
      A global analysis program for slender marine structures. :func:`~cablexsec.read_riflex_ascii`
      reads its ASCII time histories, with positional columns and a declared tension kind and
      curvature convention.

   rig test
      A combined tension and cyclic bending test of a specimen clamped at both ends.
      :func:`~cablexsec.simulate_rig_test` predicts the armour strain-gauge readings, including
      the anchored-wire effect at the clamps, as a planning tool (:doc:`rig_test_guide`).

   RMSE
   root-mean-square error
      The square root of the mean squared difference between model and measurement. Tutorial 2
      reports the moment RMSE in the recorded order as a share of the measured range.

   rotation curvature
      The right-handed curvature vector ``t × dt/ds`` on the section axes that global programs
      report, with fibre strain ``k_x y − k_y x``. The readers convert it to the
      :term:`strain gradient` ``(−k_y, k_x)``; bend moments map to ``(−M_y, M_x)``.

   Ruiz parameter
      A fretting-fatigue index formed from the local surface shear traction and slip and, in its
      second form, the local tangential stress (Ruiz et al. 1984). CableXSec does not compute it;
      its :term:`shear–slip product` is a different indicator.

   S–N curve
      The number of cycles to failure ``N`` against stress range ``Δσ``, usually ``N = C Δσ^(−m)``
      with one or two slopes, a knee and a cut-off. :class:`~cablexsec.SNcurve` holds the slopes,
      knee, threshold, test stress ratio ``R_ratio``, environment (seawater or in air), source and
      an optional thickness exponent; ``bulk_sn`` and ``fretting_sn`` select curves by name.

   sag bend
   hog bend
      The lower curvature peak of a lazy-wave cable below the buoyant arch (sag) and the upper
      peak over the arch (hog). In the readers' convention a sag bend has a negative ``y``
      rotation curvature and is in tension at ``φ = 0``, on the downward side of the section.

   scatter diagram
      A table of sea states with their probabilities of occurrence. Each sea state, or seed of
      it, becomes a :class:`~cablexsec.FatigueLoadCase` or a :class:`~cablexsec.SeaStateRun`;
      the probabilities must cover the whole exposure and sum to one.

   SCF
   stress concentration factor
      The ratio of a local peak stress to the nominal stress. CableXSec has three uses: the
      ``termination_scf`` of a layer (:term:`termination SCF`), the ``scf`` of a corrugated sheath,
      and :class:`~cablexsec.ComponentSCF`, separate factors on the axial, friction and local
      bending parts of the wire stress, for the mean and the fluctuation.

   schema
      The Pydantic data model that a cable definition must satisfy (``cablexsec/cables/schema.py``).
      A file that is not a YAML mapping with a list of layers raises ``CableSchemaError``; a field
      that breaks the schema raises ``pydantic.ValidationError``.

   screen
      A layer round the conductor or the insulation. Semiconducting screens smooth the electric
      field; metallic screens (copper tape, wire or foil) carry charging and fault currents. A
      copper wire screen can be resolved as a ring of wires and assessed with
      :func:`~cablexsec.compute_core_screen_fatigue` (:doc:`local_analysis_guide`, Section 13.2).

   sea state
      A short-term stationary sea described by significant wave height ``Hs``, peak period
      ``Tp`` and direction. A global analysis of one sea state gives one record per section.

   sea-state batch
      :func:`~cablexsec.run_sea_state_batch` and :term:`cablexsec-batch`: read a set of CableDyn
      runs once, compute the fatigue of every requested section and weight it into the annual
      damage, with optional termination, thermal, fretting and capacity checks
      (:doc:`global_local_guide`, Section 3).

   sealed annulus
      A dry armour annulus behind an intact outer sheath. Hydrostatic pressure acts on the outer
      sheath and clamps the armour, the true wall tension loads it, and an in-air S–N curve such
      as ``oc2017_armour_in_air`` applies (:doc:`local_analysis_guide`, Section 7.2).

   secant stiffness
      The ratio of moment to curvature between the origin (or a reversal) and a chosen point of
      the loop, one bending stiffness for a curvature range. The rig-test end-rotation profile
      iterates the secant stiffness of the bending loop at the clamp amplitude.

   section
      A cross-section at a fixed position along the cable. In the global-to-local workflow a
      material section is named by its unstretched arc length from End A, for example
      ``{"hangoff": 0.25, "sag": 58.4}``.

   section frame
      The transverse axes ``e1``, ``e2`` and the tangent ``t`` of a section, right-handed with
      ``e1 × e2 = t``. Curvature components and wire azimuths are measured in it; the CableDyn
      reader builds it from the static state and transports it without twist
      (:doc:`global_local_guide`, Section 2).

   section law
      The response of one cross-section to its own load history, as if the curvature were
      uniform along the wires over about one lay length. :class:`~cablexsec.SectionLaw` wraps it
      for test simulations; near anchors and steep gradients the along-wire and termination
      models apply.

   section model
      A model of one cross-section, layer by layer and wire by wire, as opposed to a
      :term:`global line model`. CableXSec is a semi-analytical section model
      (:doc:`concepts`).

   section state
      The axisymmetric normal response of the section to tension, torque, pressure and
      temperature: extension, twist, layer stresses and interface pressures, from
      :func:`~cablexsec.solve_section_state`. It can be reused in bending, component stress and
      bulk fatigue (``section_state=``, ``section_states=``).

   section-hour
      One hour of record at one section, the unit of the run-time figures (for example 0.76 s
      per section-hour for 118 wires and 32 fibres on one thread).

   seed
      One random realisation of a sea state. Seeds of one condition share its probability, and
      their damage is combined only at identical wire and fibre points.

   semiconducting screen
      A carbon-loaded polymer layer on the conductor and on the insulation that smooths the
      electric field. In the bundled cables it is part of the core layers; unlisted gaps inside
      a core are filled with the softer neighbouring material in the solved radial state.

   set-back
      The straight, frictional length between the point where the armour is anchored inside a
      fitting and the clamp face (``setback_m``, ``clamp_setback_m``). The near-clamp prediction
      depends strongly on it.

   SHA-256 digest
      A cryptographic hash of a file's bytes. The batch manifest, project files and benchmark
      sources record digests so that a changed input can be detected; a digest is a check, not a
      signature.

   shear coefficient
   Yun coefficient
      The factor ``ξ`` (0 < ξ ≤ 1) by which the stick contribution of an armour layer is reduced
      for the shear compliance of the polymer layers that carry the wire forces (Yun et al.
      2020); ``xi_per_layer`` of :func:`~cablexsec.compute_stiffness`. The stick stiffness of a
      wire is ``k = ξ E cos²α``; ξ = 1 is the plane-section limit.

   shear–slip product
      The contact indicator ``Φ = μ p0 δ*``, the nominal frictional shear times the slip
      amplitude at an armour crossing. It ranks wires and compares designs; it is not a fretting
      life and not the :term:`Ruiz parameter`.

   sheath
      An extruded polymer tube: the inner sheath over the cores, the outer sheath (oversheath)
      over the armour, and core sheaths round each power core. A ``cylindrical`` layer; a metal
      tube is a :term:`metallic sheath`.

   sheave
      A grooved wheel or chute over which a cable is pulled during installation.
      :func:`~cablexsec.compute_sheave` gives the curvature ``2/D``, the groove pressures, the
      run-on contact and the armour stresses and damage of one passage
      (:doc:`local_analysis_guide`, Section 8.2).

   signed curvature components
      The two curvature components with their signs, which keep the reversals and the rotation
      of the bending direction that a magnitude loses. CableDyn 0.4.0 writes them as ``CurvX``
      and ``CurvY``, a vector that points to the centre of curvature, so the strain gradient is
      ``(−CurvX, −CurvY)`` (:doc:`global_local_guide`, Section 2.1).

   significant wave height
   peak period
      ``Hs``, the mean height of the highest third of the waves, and ``Tp``, the period of the
      spectral peak. They define a :term:`sea state`.

   slip
      Relative sliding of a wire along its neighbours once the friction stress reaches its cap.
      A slipping armour wire keeps the friction stress of its cap and adds only its own bending
      stiffness.

   slip amplitude
      Half the largest chord of the slip trajectory of a contact over a cycle, ``δ*``. The
      fretting regime compares it with the :term:`transition amplitude`. The kinematic slip of
      free wires used by :func:`~cablexsec.compute_fretting` is an upper estimate of it.

   slip front
      The boundary between sticking and slipping wires round an armour layer. It starts at the
      neutral axis at slip onset and spreads to the extreme fibres as the curvature grows
      (Papailiou slip-front law).

   slip length
      The length ``L_s = a P/2π`` over which a wire slips at curvature ratio ``a = κ/κ_c``, with
      pitch ``P``. It sets the clamp boundary layer of an anchored wire
      (:doc:`rig_test_guide`, Section 3).

   slip onset curvature
      The curvature ``κ_on`` (``κ_c`` in the termination and rig guides) at which the first wire,
      on the neutral axis, starts to slip. It is proportional to the friction coefficient and the
      contact pressure; ``slip_onset_curvature`` of ``cablexsec.mechanics.papailiou`` and
      ``armour_slip_curvatures`` return it.

   solved radial contact state
      The elastic contact problem of the whole section, three-core bundle included
      (``contact_model="solved"``, :func:`~cablexsec.solve_cable_radial_state`): unilateral,
      frictionless contacts, the declared clamping carried as residual wire stress, tension,
      pressure, twist and temperature (:doc:`radial_model`).

   spring–slider
   Jenkin element
      An elastic spring in series with a Coulomb slider: elastic until its force reaches a cap,
      then sliding at that force. Each armour wire of the default friction model is one; the
      transverse slip model uses one per wire.

   stale record
      An :term:`analysis record` whose stored :term:`fingerprint` no longer matches its current
      inputs or engine. ``project.is_stale(record)`` tests it; a stale record is kept for
      comparison and the analysis is run again for a current one.

   stick
      The state in which a wire or contact does not slide. An armour wire sticks below its
      friction cap and follows the bending of the section with the stick stiffness
      ``k = ξ E cos²α``; in a fretting assessment stick means no slip is possible (infinite
      friction, zero amplitude or an unloaded face).

   strain gauge
      A sensor bonded to a surface that measures strain, read in microstrain (µε, 10⁻⁶).
      :func:`~cablexsec.simulate_rig_test` predicts the readings at
      :class:`~cablexsec.GaugeLocation` points on the outward faces of the armour wires and
      recommends gauge positions.

   strain gradient
   axial-strain gradient
      CableXSec's curvature convention ``(kx, ky)``: a fibre at ``(x, y) = r (cos φ, sin φ)`` has
      the bending strain ``kx x + ky y``, so the vector points to the most tensile fibre
      (:doc:`concepts`, curvature and force conventions).

   strain life
      Fatigue life as a function of strain amplitude (Coffin–Manson–Basquin), with the SWT or
      Morrow mean-strain correction and Masing stresses on the cyclic curve. Used for sheaths,
      foils, tapes and copper wires that may yield; constants are registered with
      :func:`~cablexsec.register_strain_life` (:doc:`local_analysis_guide`, Section 13.3).

   stranded conductor
      A conductor of helical layers (rings) of round wires. ``strand_layers`` resolves the rings;
      the stick–slip of each ring sets the conductor fatigue and the laid-core friction.

   structural layer
      A layer that contributes to the bending stiffness and is a fatigue site (``structural:
      true``, the default). A non-structural layer still transmits pressure.

   subsurface stress
      The elastic stress tensor below a contact surface. :func:`~cablexsec.compute_hertz_axis_stress`
      gives it on the axis of a Hertz contact and :func:`~cablexsec.compute_hertz_subsurface_stress`
      at any point under prescribed Hertz-shaped normal and shear tractions.

   SWT parameter
      The Smith–Watson–Topper fatigue parameter ``σ_max ε_a``, the peak normal stress on a plane
      times the strain amplitude normal to it. It is a mean-stress correction of the bulk count
      (``mean_stress="swt"``), the driver of the strain-life and trellis routes, and the
      indicator of the :term:`critical plane` search of :func:`~cablexsec.compute_swt_cycle`.

   tape
      A helically wound strip (copper, polymer or laminate) over a layer, declared by a ``tape``
      winding with width, angle and source. :func:`~cablexsec.tape_geometry` gives its pitch, gap
      and coverage; :func:`~cablexsec.compute_tape_fatigue` its strain and strain-life damage.

   tensile armour
      Armour designed to carry axial tension, the term of flexible-pipe practice for the helical
      wire layers that CableXSec analyses; pressure armour is outside its scope.

   tension kind
      The force a history holds: ``"true"`` (true wall tension), ``"effective"`` or
      ``"unspecified"`` (``tension_kind`` of a :term:`load case series`). The CSV reader infers it
      from the heading; fatigue routes refuse ``"unspecified"`` and convert ``"effective"`` only
      on request.

   termination
      The end of a cable section at an end fitting or a clamped bend-stiffener root, where the
      armour is anchored. See :term:`termination SCF` and :term:`termination correction`.

   termination correction
      The optional replacement of the sectional friction stress near an anchor by the
      anchored-wire solution along the curvature history (``termination=``
      :class:`~cablexsec.TerminationSpec` in :func:`~cablexsec.compute_fatigue`). It is a model
      estimate: against the Sævik (2011) measurements it over-predicts
      (:doc:`local_analysis_guide`, Section 5, Step 6).

   termination SCF
      A stress-range factor at an end fitting declared on a layer (``termination_scf``; 1.0 means
      none). ``damage_basis="bulk_with_termination"`` of
      :func:`~cablexsec.compute_long_term_fatigue` accumulates the amplified field.

   test arrangement
      The way a bending test applies load: three-point bending (moment rising linearly from the
      supports), four-point bending (uniform moment between the loads) or a cantilever.
      :func:`~cablexsec.simulate_bending_test` with :class:`~cablexsec.ThreePoint`,
      :class:`~cablexsec.FourPoint` or :class:`~cablexsec.Cantilever` compares a model with a test
      through its own arrangement and reduction (:doc:`local_analysis_guide`, Section 6.2).

   thermal network
      A radial chain of thermal resistances and heat capacities (an RC ladder) through the cable
      layers, integrated in time by TR-BDF2 (``cx.thermal_network``). It gives the transient
      temperatures of :class:`~cablexsec.ThermalHistory` from a current history.

   thermal resistance
      The temperature rise per unit heat flow through a layer [K m/W]. In the IEC 60287 ladder
      ``T1`` lies between conductor and metallic sheath or screen, ``T2`` and ``T3`` outside it, and
      ``T4`` is the external resistance of the surroundings (``T4_external_Km_W``).

   three-point bending
      See :term:`test arrangement`.

   tier
      The source class of a validation gate: Tier 1 is a closed form, an invariant or a second
      CableXSec code path; Tier 2 is a public measurement or published model. A tier is not a
      quality ranking (:doc:`validation_policy`).

   time-history CSV
      A CSV export of a global analysis with one header row of named columns.
      :func:`~cablexsec.read_time_history_csv` selects columns by whole words of their heading and
      converts units only through its scale arguments.

   torque balance
      The cancelling of the torques of contra-helical layers under tension.
      :func:`~cablexsec.torque_balance` gives the end torque with twist restrained, the twist of
      a free end and the layer torques; :func:`~cablexsec.design_torque_balance` proposes the lay
      angle, wire count or wire diameter that zeroes the torque (:doc:`local_analysis_guide`,
      Section 12).

   touchdown
      The region where a cable meets the seabed. Its effective tension is low and the
      touchdown point moves with the floater, so it can see armour compression (:doc:`uls_guide`).

   transition amplitude
      The slip amplitude ``δ_t`` that separates partial from gross slip at a contact. For a
      Mindlin point contact ``δ_t = 3μP/(2K_t)`` (Fouvry et al. 1995); for a wire on polymer it
      follows from the support compliance of the section law.

   transverse slip
      Slip of a wire across its own direction on the cable surface, towards the geodesic. The
      default kinematics have none; ``transverse_ratio`` with ``transverse_map="geodesic"`` adds it
      as a bounded sensitivity, and :class:`~cablexsec.TransverseSlipModel` solves its
      equilibrium with friction (:doc:`local_analysis_guide`, Sections 6.4 and 14).

   trellis contact
      The point contacts between wires of neighbouring conductor rings laid in opposite
      directions. The opt-in trellis term of :func:`~cablexsec.compute_conductor_fatigue` assesses
      them with the :term:`critical distance` rule; the legacy surface SWT rule is not validated.

   true wall tension
      The axial force carried by the cable wall itself, ``T_w = T_e − p_o A_o`` for a cable with no
      bore. Fatigue of a sealed annulus needs it; ``series.with_true_tension(...)`` converts an
      effective-tension history.

   ULS
   ultimate limit state
      The design check against failure under extreme load: yield, buckling, rupture.
      :func:`~cablexsec.compute_capacity`, :func:`~cablexsec.build_capacity_envelope`,
      :func:`~cablexsec.compression_screen`, coiling and sheave checks serve it (:doc:`uls_guide`).

   umbilical
      A bundle of steel tubes, hoses, power and signal cables with helical armour that supplies
      subsea equipment. Their tubes are outside the scope of CableXSec; umbilical axial stiffness
      tests appear in the register as BM4.

   utilisation
      Load over capacity; one means the limit is reached. The capacity check reports the
      utilisation history of a global record; the design utilisation divides it by the
      :term:`design utilisation factor`.

   validation
      A comparison with measured data for a stated construction, loading path and output
      quantity, which shows whether the model describes a physical cable. It differs from
      :term:`verification` (:doc:`validation_policy`).

   validation status
      The status of a scorecard row: PASS (target met), FAIL (target not met; reported, not
      loosened), REVIEW, PARTIAL, NOT VALIDATED, NOT COMPARABLE or INFO (:doc:`validation_policy`).

   verification
      A check that the equations are solved correctly, by closed forms, invariants, convergence
      and agreement between two code paths. It says nothing about whether the equations describe
      a real cable.

   von Mises stress
      An equivalent stress that combines the stress components into one value for comparison
      with the uniaxial yield stress. It is used for the corrugated-sheath first-yield check and
      the Hertz subsurface stress.

   warning classes
      Warnings issued when a run succeeds but rests on an assumption or leaves its validated
      range: :term:`ExtrapolationWarning`, :term:`MissingClampingStateWarning`,
      ``UnmodelledPolymerSofteningWarning``, ``UnevaluatedFatigueCurveWarning``,
      ``UnvalidatedModelWarning``, ``TransverseMapWarning`` and ``FrictionModelFallbackWarning``
      (``friction_model="auto"`` kept the direction-tied law on a non-planar path); and the
      ``FutureWarning``
      ``DirectionTiedFrictionWarning`` of a default that changes in the next release.

   wear
      Loss of material from a contact by sliding. :func:`~cablexsec.compute_fretting` computes the
      worn volume from the dissipated energy, ``V = α Σ W``, with an energy coefficient per material
      pair; the steel-on-polymer coefficients are assumptions, and the reported depth rate is that
      of the unworn contact.

   wet design
   dry design
      A wet design lets water reach the cores and armour and has no radial metallic water barrier;
      a dry design must contain one (lead or aluminium sheath). Declared as
      ``water_ingress.design`` (CIGRE TB 610); it does not change any analysis default.

   winding stress
      The residual wire stress retained in an armour layer from winding during manufacture
      (``armour_winding_stress_Pa``). The bounded chain converts it to a pressure jump
      (Féret and Bournazel 1987, Eq. 5); the solved radial state carries it as residual wire
      stress.

   wire azimuth
      The angle ``φ`` of a wire round the section, from the section ``x`` axis (``e1``) towards
      ``y``; wire ``k`` of ``n`` sits at ``360 k / n`` degrees. Reported as ``phi_critical_deg``.

   Witz–Tan stiffness matrix
      The coupled axial–torsional stiffness of a helically armoured section, with terms
      ``K_ee`` (extension), ``K_et`` (tension–twist coupling) and ``K_tt`` (torsion) (Witz and Tan
      1992). :func:`~cablexsec.compute_stiffness` and :func:`~cablexsec.xsec_stiffness` report it;
      ``K_et / K_ee`` is the torque per unit tension with twist restrained.

   XLPE
      Cross-linked polyethylene, the usual insulation of power cores. It has a registered
      modulus–temperature law (``xlpe_qiao2025``) and the parameters of the electro-thermal ageing
      model.

   YAML
   JSON
   CSV
      Text file formats: YAML for cable definitions and geometry overrides; JSON for the batch
      specification, project files and exported bending laws; CSV for load histories and bending
      tables. Layouts are given in :doc:`file_formats`.

Symbols
-------

Symbols as the guides write them. Where a symbol has more than one meaning, both are given.

.. glossary::

   a
      The curvature ratio ``κ/κ_c`` of an armour layer, used for the slip length and the clamp law
      of anchored wires (:doc:`rig_test_guide`).

   D
      The Miner damage of a record or per year (``D_year``); also a sheave, tank or bore diameter
      where the context says so.

   GJ
      The torsional stiffness [N m²]. ``gj_Nm2`` of :func:`~cablexsec.compute_stiffness` is the
      contribution of the helical lay under extension, without the independent torsion of the
      components.

   K_t
      The tangential stiffness of a Hertz contact [N/m], used by the Mindlin slip model and the
      transition amplitude.

   ℓ_x
      The length of the :term:`clamp boundary layer`, ``½ √(L_s P)``.

   m
      The slope of an S–N curve, ``N ∝ Δσ^(−m)``.

   n
      The number of wires of a helical layer (``n_wires``); ``n_i`` also counts the cycles of
      range ``Δσ_i``.

   p0
      The peak pressure of a Hertz contact. In the tutorials ``p0`` also names the initial
      pressure of the ``initial_state`` block; ``P0`` is the clamping jump.

   P0
      The pressure jump across an armour layer from its manufacturing clamping
      (``initial_contact_pressure_Pa``). The jumps accumulate from the outside in: two layers each
      with ``P0`` = 2 MPa have ``P1 + P2`` of 6 MPa (inner) and 2 MPa (outer).

   P1 + P2
      The friction-setting contact pressure of an armour layer: the pressures on its inner and
      outer faces, referred to the helix radius. ``P·2πr/n`` is the line load per wire.

   r
      The :term:`helix radius` of a wire (``R`` in some guides); in the stick stress ``k r κ c``
      the lever arm of the wire from the cable axis.

   T_e, T_w
      The :term:`effective tension` and the :term:`true wall tension`, related by
      ``T_w = T_e − p_o A_o`` with the external pressure ``p_o`` on the outer area ``A_o``.

   T_ref
      The stress-free reference temperature of the free thermal strain ``α (T − T_ref)``. The
      thermal histories default it to the ambient of their network, the cable as installed.

   α
      The :term:`lay angle` of a helical layer, from the cable axis; also the thermal expansion
      coefficient (``alpha_T_K``) and the energy wear coefficient [m³/J] in their contexts.

   β
      The lay angle of the laid power cores (:term:`core lay`) or of a tape.

   δ*, δ_t
      The :term:`slip amplitude` of a contact and its :term:`transition amplitude`.

   η
      The :term:`design utilisation factor`; also the transverse slip fraction (0 loxodrome to 1
      geodesic) of the tendon and transverse-slip models.

   κ
      Curvature [1/m]; ``kx``, ``ky`` are the components of the :term:`strain gradient`.

   κ_on, κ_c
      The :term:`slip onset curvature` of an armour layer (``κ_c`` in the termination, rig and
      conductor sections).

   μ
      The :term:`friction coefficient`.

   ξ
      The :term:`shear coefficient` of an armour layer (Yun et al. 2020).

   σ_T
      The thermal wire stress, the solved wire stress at a temperature minus the reference at
      the same tension.

   φ
      The :term:`wire azimuth` round the section, measured from ``e1`` (or the section ``x``
      axis) towards ``e2``.

   ψ
      The argument of the friction cap, ``ψ = asin c`` with ``c = |cos(φ − φ_b)|``; zero at the
      neutral axis and ``π/2`` at the extreme fibre.
