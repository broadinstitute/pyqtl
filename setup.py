import os
import re
from setuptools import setup, find_packages
with open("qtl/__init__.py") as reader:
    __version__ = re.search(
        r'__version__ ?= ?[\'\"]([\w.]+)[\'\"]',
        reader.read()
    ).group(1)
_README           = os.path.join(os.path.dirname(__file__), 'README.md')
_LONG_DESCRIPTION = open(_README).read()

# Setup information
setup(
    name = 'qtl',
    version = __version__,
    packages = find_packages(),
    description = 'Utilities for analyzing and visualizing QTL data',
    author = 'Francois Aguet (Broad Institute)',
    author_email = 'francois@broadinstitute.org',
    long_description = _LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    install_requires = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'pyBigWig',
        'bx-python',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
