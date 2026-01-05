"""
ANDES Documentation - New Structure
Built with Sphinx and MyST for easier authoring.
"""

import os
import andes

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
nb_execution_mode = 'off'             # Don't execute notebooks (use pre-executed)
nb_execution_timeout = 300            # 5 min timeout per cell
nb_execution_raise_on_error = False   # Don't fail build on notebook errors
nb_execution_excludepatterns = [
    'verification/*',                 # Skip verification notebooks
    'tutorials/*',                    # Skip tutorial notebooks
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
    'doc_path': 'docs_new/source',
}

html_static_path = ['_static']
html_favicon = '../../docs/source/images/curent.ico'
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

# Generate model reference (reuse existing script logic)
if not os.path.isfile('reference/models/_generated'):
    os.makedirs('reference/models', exist_ok=True)

    ss = andes.prepare(nomp=True)

    # Generate model reference index
    out = """# Model Reference

Use the navigation to browse models by group.

"""
    out += ss.supported_models(export='rest')
    out += '\n\n```{toctree}\n:maxdepth: 2\n:hidden:\n\n'

    for group in ss.groups.values():
        out += f'{group.class_name}\n'

    out += '```\n'

    with open('reference/models/index.md', 'w') as f:
        f.write(out)

    # Generate individual group files
    for group in ss.groups.values():
        with open(f'reference/models/{group.class_name}.rst', 'w') as f:
            f.write(group.doc_all(export='rest'))

    # Mark as generated
    with open('reference/models/_generated', 'w') as f:
        f.write('generated')

# Generate config reference
if not os.path.isfile('reference/config.rst'):
    ss = andes.prepare(nomp=True)

    out = '.. _configref:\n\n'
    out += '****************\n'
    out += 'Config Reference\n'
    out += '****************\n\n'
    out += ss.config.doc(export='rest', target=True, symbol=False)

    for r in ss.routines.values():
        out += r.config.doc(export='rest', target=True, symbol=False)

    with open('reference/config.rst', 'w') as f:
        f.write(out)

# Disable smartquotes for double dashes
smartquotes = False
