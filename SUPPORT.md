<!-- SPDX-License-Identifier: Apache-2.0 -->

# Getting help

CableXSec is maintained by SMI Lab, Inha University, on a best-effort basis. All
support is through public GitHub Issues, so that answers help the next user too.

1. **Read the manual first.** The [manual](https://cablexsec.readthedocs.io) covers
   installation, the cable definition, the local analysis and global-to-local guides,
   the benchmarks and the Python API.
2. **Search existing issues**, open and closed, at
   <https://github.com/SMI-Lab-Inha/CableXSec/issues>.
3. **Open an issue** with the matching template:

   | You have | Template |
   |---|---|
   | A crash, a wrong result, or a valid input that is rejected | Bug report |
   | A capability you need | Feature request |
   | A question about modelling, inputs, or outputs | Question |

   Attach the smallest cable YAML and script that show the problem and every file they
   read, and state the CableXSec version
   (`python -c "import cablexsec; print(cablexsec.__version__)"`).

Security vulnerabilities are not reported in public; follow
[SECURITY.md](SECURITY.md). For questions about citing CableXSec or collaboration,
contact the maintainer at <jaehoon.seo@inha.ac.kr>.

The maintainers cannot review proprietary cable designs or give design advice for a
specific project; CableXSec users remain responsible for their engineering decisions.
Do not attach cable data that you are not permitted to publish.
