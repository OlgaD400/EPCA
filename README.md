# Ensemble Principal Component Analysis
Efficient representations of data are essential for processing, exploration, and human understanding, and _Principal Component Analysis_ (PCA) is one of the most common dimensionality reduction techniques used for the analysis of large, multivariate datasets today. Two well-known limitations of the method include sensitivity to outliers and noise and no clear methodology for the uncertainty quantification of the principal components or their associated explained variances. Whereas previous work has focused on each of these problems individually, we propose a scalable method called Ensemble PCA (EPCA) that addresses them simultaneously for data which has an inherently low-rank structure. EPCA combines boostrapped PCA with k-means cluster analysis to handle challenges associated with sign-ambiguity and the re-ordering of components in the PCA subsamples. EPCA provides a noise-resistant extension of PCA that lends itself naturally to uncertainty quantification. We test EPCA on data corrupted with white noise, sparse noise, and outliers against both classical PCA and Robust PCA (RPCA) and show that EPCA performs competitively across different noise scenarios, with a clear advantage on datasets containing outliers and orders of magnitude reduction in computational cost compared to RPCA.

## Repository Description 
This repository implements Ensemble Principal Component Analysis (EPCA) and provides example notebooks to demonstrate the usage of the method. 

## Files 
*EPCA/EPCA.py: Implementation of Ensemble Principal Component Analysis (EPCA).
* helper_functions.py: Contains all functions for adding noise to datasets, running EPCA, PCA, and RPCA on said noisy data, and writing performance results to an output file.
* plotting_functions.py: Contains all functions for generating figures from the results of various experiments.
* tests/tests.py: Contains tests for EPCA.

## Scripts
* mnist_script.py: Script for running experiments on digits of the MNIST dataset.
* seasurfacetemp_script.py: Script for running experiments on the sea surface temperature dataset.
* sklearn_data_script.py: Script for running experiments on select sklearn datasets.
* wave_script.py: Script for running experiments on synthetic wave data.


## Notebooks
* notebooks/Compare_Performance_Summary_Plots.ipynb: Executes functions used to load data and recreate plots from the paper. 
* notebooks/MNIST.ipynb: Walks a user through running and comparing the performance of EPCA, PCA, and RPCA on digits of the MNIST dataset.
* notebooks/SeaSurfaceTemp.ipynb: Walks a user through running and comparing the performance of EPCA, PCA, and RPCA on the sea surface temperature dataset.
* notebooks/SklearnData.ipynb: Walks a user through running and comparing the performance of EPCA, PCA, and RPCA on various sklearn datasets.
* notebooks/WaveData.ipynb: Walks a user through running and comparing the performance of EPCA, PCA, and RPCA on a synthetic wave dataset.
