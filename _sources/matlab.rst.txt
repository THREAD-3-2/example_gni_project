.. _matlab:

MATLAB code
===========

This primitive example just shows how to document MATLAB code with Sphinx
using the `matlabdomain extension <https://github.com/sphinx-contrib/matlabdomain/blob/master/README.rst>`_.


MyModule
--------

.. mat:automodule:: MyModule

:mod:`MyModule` module contains a function
which implements the explicit Euler method:
    
.. mat:autofunction:: MyModule.Forward_Euler

The module further contains an example script:

.. mat:autoscript:: MyModule.main
