"""
ANDES Documentation - New Structure
Built with Sphinx and MyST for easier authoring.
"""

import os
from pathlib import Path

import andes

# Absolute path to this source directory (works regardless of CWD or folder name)
SRCDIR = Path(__file__).parent.resolve()

# -- Project information -----------------------------------------------------

project = 'ANDES'
copyright = '2026, Hantao Cui'
author = 'Hantao Cui'
version = andes.__version__
release = andes.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_nb',                    # MyST Markdown + Jupyter notebooks
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_design',              # Modern cards, tabs, grids
    'sphinx_copybutton',          # Copy button for code blocks
    'sphinx_selective_exclude.eager_only',  # Process {only} during parsing
    'numpydoc',
]

# MyST configuration
myst_enable_extensions = [
    'colon_fence',      # ::: directive syntax
    'deflist',          # Definition lists
    'dollarmath',       # $math$ and $$math$$
    'fieldlist',        # Field lists
    'substitution',     # {{variable}} substitution
    'tasklist',         # - [ ] task lists
]

myst_heading_anchors = 3  # Generate anchors for h1-h3

# Notebook execution
nb_execution_mode = 'auto'            # Execute notebooks without outputs, skip those with outputs
nb_execution_timeout = 300            # 5 min timeout per cell
nb_execution_raise_on_error = True    # Fail build on notebook errors (catch issues in CI)
nb_execution_excludepatterns = [
    'verification/*',                 # Skip verification notebooks (long-running)
]
nb_merge_streams = True               # Merge stdout/stderr into single output

# AutoDoc
autosummary_generate = True
numpydoc_show_class_members = False

# Source files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', '_templates']
templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'use_edit_page_button': True,
    'show_toc_level': 2,
    'navigation_with_keys': True,
    'show_nav_level': 2,
    'navbar_align': 'left',
    'navbar_center': ['navbar-nav'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page'],
    'icon_links': [
        {
            'name': 'PDF Manual',
            'url': 'https://docs.andes.app/_/downloads/en/stable/pdf/',
            'icon': 'fa-solid fa-file-pdf',
        },
        {
            'name': 'GitHub',
            'url': 'https://github.com/CURENT/andes',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/andes/',
            'icon': 'fa-solid fa-box',
        },
    ],
}

html_context = {
    'github_url': 'https://github.com',
    'github_user': 'CURENT',
    'github_repo': 'andes',
    'github_version': 'master',
    'doc_path': 'docs/source',
}

html_static_path = ['_static']
html_logo = 'images/andes_logo.svg'
html_favicon = 'images/curent.ico'
html_title = f'ANDES {version}'

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Generate model reference ------------------------------

# Use absolute paths to avoid issues with different working directories
_models_dir = SRCDIR / 'reference' / 'models'
_models_marker = _models_dir / '_generated'
_config_file = SRCDIR / 'reference' / 'config.rst'

# Generate reference docs (only if marker files don't exist)
_needs_models = not _models_marker.exists()
_needs_config = not _config_file.exists()

if _needs_models or _needs_config:
    # Only call prepare() once for all generation
    ss = andes.prepare(nomp=True)

    if _needs_models:
        _models_dir.mkdir(parents=True, exist_ok=True)

        # Generate model reference index
        out = """# Model Reference

Use the navigation to browse models by group.

```{eval-rst}
"""
        out += ss.supported_models(export='rest')
        out += '```'
        out += '\n\n```{toctree}\n:maxdepth: 2\n:hidden:\n\n'

        for group in ss.groups.values():
            out += f'{group.class_name}\n'

        out += '```\n'

        (_models_dir / 'index.md').write_text(out)

        # Generate individual group files
        for group in ss.groups.values():
            (_models_dir / f'{group.class_name}.rst').write_text(
                group.doc_all(export='rest')
            )

        # Mark as generated
        _models_marker.write_text('generated')

    if _needs_config:
        out = '.. _configref:\n\n'
        out += '****************\n'
        out += 'Config Reference\n'
        out += '****************\n\n'
        out += ss.config.doc(export='rest', target=True, symbol=False)

        for r in ss.routines.values():
            out += r.config.doc(export='rest', target=True, symbol=False)

        _config_file.write_text(out)

# Disable smartquotes for double dashes
smartquotes = False

# -- Options for LaTeX output ------------------------------------------------

# Use xelatex for Unicode support
latex_engine = "xelatex"

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
\setcounter{tocdepth}{3}
\setlength{\headheight}{13.6pt}
\addtolength{\topmargin}{-1.6pt}
''',  # Show chapters, sections, subsections in TOC; fix fancyhdr warning
}

latex_documents = [
    (master_doc, 'andes.tex', 'ANDES Manual', 'Hantao Cui', 'manual'),
]
