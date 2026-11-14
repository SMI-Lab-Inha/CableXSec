.. SPDX-License-Identifier: Apache-2.0

Command line
============

``cablexsec-batch``
-------------------

Computes the weighted annual armour damage from a set of CableDyn sea-state runs.

.. code-block:: powershell

   cablexsec-batch batch.json --output-directory results\batch1 [--workers N]

``batch.json`` names the cable definition (a YAML path, or ``bundled:<name>`` for a cable
shipped with the package), the CableDyn runs with their probabilities and retained windows,
the material sections and the options; :func:`cablexsec.analysis.sea_state_batch.load_batch_spec`
documents every key. The output directory must not exist. It receives ``manifest.json`` (input
digests, checks, options and results), the annual fixed-fibre damage of every section and the
section history of every run. ``--workers`` overrides the number of worker processes of the
specification; each worker runs with one thread (``OMP_NUM_THREADS`` and the related variables
default to ``1``).

The console prints one line per section with the arc length, the annual damage, the life in
years and the critical layer, followed by the SHA-256 digest of the inputs. The
:doc:`global_local_guide` explains the specification, the sign conventions of the CableDyn
section tables and the interpretation of the results.

Validation and reproduction scripts
-----------------------------------

The scripts in ``validation/scripts/`` are run with Python from the repository root, for
example:

.. code-block:: powershell

   python validation/scripts/run_validation.py --check
   python validation/scripts/check_out_von_morgen.py

They are described in the
`validation README <https://github.com/SMI-Lab-Inha/CableXSec/blob/main/validation/README.md>`_.

.. autofunction:: cablexsec.cli.batch_main
