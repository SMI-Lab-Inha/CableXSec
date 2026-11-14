# Configuration file for the Sphinx documentation builder.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
#
# Build locally (as in DEVELOPMENT.md):
#   python -m sphinx -W --keep-going -b html doc doc/_build/html
# (or `make -C doc html`).  Read the Docs uses ../.readthedocs.yaml.

import sys
import tomllib
from pathlib import Path

# Make the package importable for autodoc in local builds.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# -- Project information -----------------------------------------------------

project = "CableXSec"
author = "Jae Hoon Seo"
project_copyright = "2026, Jae Hoon Seo, SMI Lab, Inha University"

# pyproject.toml is the single source of the release number.
with (_REPO_ROOT / "pyproject.toml").open("rb") as _handle:
    release = tomllib.load(_handle)["project"]["version"]
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
]

# The manual includes the repository's Markdown pages (VALIDATION.md, CHANGELOG.md,
# the analysis guides) verbatim through MyST.
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Included Markdown must use absolute GitHub URLs for repository files outside doc/.
suppress_warnings = ["autosectionlabel.*", "myst.header"]

# `make linkcheck`: these URLs become live only when the repository is published.
linkcheck_ignore = [
    r"https://cablexsec\.readthedocs\.io/.*",
    r"https://github\.com/SMI-Lab-Inha/CableXSec.*",
]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "CableXSec documentation"
html_short_title = "CableXSec"
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = False

html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
    "style_external_links": True,
    "prev_next_buttons_location": "both",
}

html_context = {
    "display_github": True,
    "github_user": "SMI-Lab-Inha",
    "github_repo": "CableXSec",
    "github_version": "main",
    "conf_py_path": "/doc/",
}

autodoc_typehints = "none"
autodoc_member_order = "bysource"

# -- LaTeX / PDF output ------------------------------------------------------

latex_engine = "xelatex"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
}
latex_documents = [
    ("index", "CableXSec.tex", "CableXSec documentation", author, "manual"),
]

# -- EPUB output -------------------------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = project_copyright
epub_show_urls = "footnote"
epub_exclude_files = [".nojekyll"]

copybutton_prompt_text = r">>> |\.\.\. |\$ |PS [^>]*> "
copybutton_prompt_is_regexp = True
