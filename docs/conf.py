# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'example_gni_project'
copyright = '2022, Markus Lohmayer'
author = 'Markus Lohmayer'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

master_doc = "index"
latex_title = f"Example GNI Project Documentation"


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # automatically check that examples work with `make doctest`
    'sphinx.ext.doctest',
    # automatically extract in-line documentation from your (Python) code
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    # add ability to automatically extract in-line documentation from MATLAB code
    'sphinxcontrib.matlab',
]

# specify the source folder of the MATLAB code
# For a pure MATLAB project, you probably want to use `src` as the name.
matlab_src_dir = "../matlab_src/"
# For a pure MATLAB project, you can specify `mat` as the primary domain to save some typing,
# e.g. `autoclass` instead of `mat:autoclass` (see matlab.rst file).
# primary_domain = 'mat'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# configure LaTeX output
latex_documents = [(master_doc, f"{project}-{release}.tex", latex_title, author, "manual")]
latex_elements = {
'papersize': 'a4paper',
'pointsize': '12pt',
}
