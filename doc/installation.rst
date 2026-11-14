.. SPDX-License-Identifier: Apache-2.0

Installation
============

CableXSec is a pure-Python package for Python 3.11 and 3.12 on Linux, macOS and Windows. It
is distributed through its GitHub repository; it is not published on a package index.

From a release
--------------

Download ``cablexsec-0.1.0-py3-none-any.whl`` and ``SHA256SUMS.txt`` from the
`v0.1.0 release <https://github.com/SMI-Lab-Inha/CableXSec/releases/tag/v0.1.0>`_, compare
the checksum, and install the wheel with its optional extras:

.. code-block:: powershell

   Get-FileHash cablexsec-0.1.0-py3-none-any.whl -Algorithm SHA256
   python -m pip install "cablexsec-0.1.0-py3-none-any.whl[fast]"

From the repository
-------------------

.. code-block:: powershell

   python -m pip install "cablexsec[fast] @ git+https://github.com/SMI-Lab-Inha/CableXSec.git@v0.1.0"

or, for development, clone the repository and install it in editable mode (see
:doc:`development`):

.. code-block:: powershell

   git clone https://github.com/SMI-Lab-Inha/CableXSec.git
   cd CableXSec
   python -m pip install -e ".[dev,fast,contact]"

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Extra
     - Adds
   * - ``fast``
     - ``numba``: compiled per-wire slip-stress and rainflow kernels. Results agree with the
       NumPy reference to floating-point round-off and rainflow counts are bitwise equal;
       ``CABLEXSEC_DISABLE_NUMBA=1`` selects the reference path.
   * - ``contact``
     - ``clarabel``: an independent conic solver used only by the cross-check script
       ``validation/scripts/concentric_conductor_bend.py`` and its test; the package's own
       contact solvers do not need it.
   * - ``dev``
     - test, lint, type-check and build tools.

Check the installation:

.. code-block:: powershell

   python -c "import cablexsec; print(cablexsec.__version__, cablexsec.__file__)"
   cablexsec-batch --help
