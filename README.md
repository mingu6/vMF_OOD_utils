# Mixture of von-Mises Fisher fitting for OOD

Contains utilities for performing OOD by fitting vMF mixture distributions to embedding vectors. OOD scores are generated using the negative log-likelihood evaluated using the fitted distributions.

## Dependencies

* [spherecluster](https://github.com/jasonlaska/spherecluster) package reproducing ["Clustering on the Unit Hypersphere using von Mises-Fisher Distributions"](http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf), Banerjee et al., JMLR 2005.
* pytorch
* scikit-learn

Simple conda environment:

```
conda create -n vMF_OOD python=3.8 scikit-learn pytorch -c pytorch
pip install spherecluster
```

## Usage
