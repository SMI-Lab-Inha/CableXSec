.. SPDX-License-Identifier: Apache-2.0

How to cite
===========

If CableXSec contributes to published work, cite the software release with the version you
used. The entry is kept in the repository's ``CITATION.cff``, which GitHub exposes through its
**Cite this repository** function. The journal article that describes the method is under
review; its reference will be added here and to ``CITATION.cff`` when it is published.

Software release
----------------

Seo, J. H. (2026). *CableXSec: Component-resolved local cross-section mechanics of dynamic
power cables* (version 0.2.1) [Computer software]. https://github.com/SMI-Lab-Inha/CableXSec

.. code-block:: bibtex

   @software{CableXSec_0_2_0,
     author  = {Seo, Jae Hoon},
     title   = {{CableXSec}: Component-resolved local cross-section mechanics of
                dynamic power cables},
     version = {0.2.1},
     year    = {2026},
     url     = {https://github.com/SMI-Lab-Inha/CableXSec},
     license = {Apache-2.0}
   }

When results depend on a particular release, quote ``cablexsec.__version__`` together with the
provenance records of the results (a saved project keeps both).

Global analysis with CableDyn
-----------------------------

Global-to-local studies that use CableDyn results should also cite CableDyn:

Seo, J. H., Lim, J., Shim, K. & Song, J. (2026). CableDyn: Curvature-resolving implicit
finite-element analysis of mooring lines and dynamic power cables for floating offshore
wind. *Ocean Engineering* 368 (Part 2), 128332.
https://doi.org/10.1016/j.oceaneng.2026.128332

Citing methods and reference data
---------------------------------

CableXSec implements published methods (for example the neutral-axis slip front of
Papailiou, the Feret–Bournazel winding pressure, Mindlin partial slip, the IEC 60287 thermal
ladder) and is compared with published data (Ménard and Cartraud 2023, Witz 1996, Sævik
2011 and others). The bundled cable definitions and benchmark data cite their sources in
each file. When a study relies on a specific method or data set, cite the original source as
well.
