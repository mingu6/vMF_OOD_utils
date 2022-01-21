# Mixture of von-Mises Fisher fitting for OOD

Contains utilities for performing OOD by fitting vMF mixture distributions to embedding vectors. OOD scores are generated using the negative log-likelihood evaluated using the fitted distributions.

## Dependencies

* [spherecluster](https://github.com/jasonlaska/spherecluster) package reproducing ["Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"](http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf), Banerjee et al., JMLR 2005. Clone the repo and install using the repo directly instead of installing off of PyPI (do NOT `pip install spherecluster`).
* pytorch
* scikit-learn==0.22 (do NOT deviate from this version!!!)

Simple conda environment:

```
conda create -n vMF_OOD python=3.8 scikit-learn=0.22 pytorch -c pytorch
conda activate vMF_OOD
git clone https://github.com/jasonlaska/spherecluster
cd spherecluster
python setup.py install
```

## Usage

See `utils.py` for an example usage. Idea is to have a fit function and a log likelihood evaluation function given a fitted model.

## TODO

Seems a bit slow right now for large dimensions and large number of observations... Could combine this with skorch to try GPU stuff?
