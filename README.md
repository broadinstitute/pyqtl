## pyQTL

pyQTL is a python module for analyzing and visualizing quantitative trait loci (QTL) data.

The following functionalities are provided:
* `qtl.annotation`: class for working with gene annotations; includes a [GTF](https://www.gencodegenes.org/pages/data_format.html) parser.
* `qtl.coloc`: Python implementation of core functions from the [R COLOC package](https://github.com/chr1swallace/coloc).
* `qtl.io`: functions for reading/writing BED and GCT files.
* `qtl.locusplot`: functions for generating LocusZoom-style regional association plots.
* `qtl.pileup`: functions for visualizing QTL effects in read pileups from, e.g., RNA-seq data.
* `qtl.plot`: plotting functions for QTLs.

### Install
You can install pyQTL using pip:
```
pip3 install qtl
```
or directly from this repository:
```
$ git clone git@github.com:broadinstitute/pyqtl.git
$ cd pyqtl
# set up virtual environment and install
$ virtualenv venv
$ source venv/bin/activate
(venv)$ pip install -e .
```
