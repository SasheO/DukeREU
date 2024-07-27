# DukeREU

## About
This directory contains code, presentation, and some other relevant files and folder used by Mezisashe Ojuba during the 2024 Pratt School of Engineering REU. The project of this REU was to develop simplified phoneme-specific mask estimation model building on the [earlier work of Kevin Chu](https://doi.org/10.1121/2.0001698). This readme describes the subfolders as well as important locations to find results, visualizations, etc. This directory contains only a checkpoint of useful code. Moreover, this directory is currently unstructured. The main code-base is complete and better documented.



## Descriptions of Sub Folders/Files

`REU Poster`: This contains the final and draft editions of the REU poster, the images used in it, and the abstract draft.

`weekly presentation`: This contains the slideshow for the weekly presentations during the course of the REU.

`saved_variables/`: This contains various saved variables useful for different computations and visualizations.

`TitanV/`: This repository contains code to develop simplified phoneme-specific mask estimation model. It is built off the work in chu.kevin folder. The readme provides a more in-depth description.

`Task 1- Simulate Audio CI Processing Pipeline.ipynb`: This is a scratch-paper jupyter notebook containing the code to simulate the cochlear implant auditory processing pipeline.

`Task 2 - umap.ipynb/`: This is scratch-paper jupyter notebook containing the code to plot umap [1] visualizations of timit dataset before and after normalization by phoneme. For more organized and well-documented umap plotting functions, see `visualizations/plot_umap.py` or `TitanV/MaskEstimationPytorch/Task_5 - presentation.py`.

`Task 3 - zmuv.ipynb`: This is scratch-paper jupyter notebook containing the code to zmuv-normalize timit dataset phoneme.

`Task 4 - build mlp.ipynb`: This is scratch-paper jupyter notebook containing the code to build and test the MLP models and architecture used in LSTMTransformModel found in `TitanV\MaskEstimationPytorch\net.py`.

`TitanV/MaskEstimationPytorch/Task_5 - presentation.py`: This contains the code to plot the UMAP visualizations, calculate the separability score, and plot the bar charts comparing STOI/SRMR scores of different model architectures. This plots the output of the experiments stored in `TitanV/MaskEstimationPytorch/exp/` folder.