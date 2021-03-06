# qFADD.py
This repository contains the python implementation of the "Quantitation of Fluorescence Accumulation after DNA Damage (Q-FADD)" Analysis Routine (https://doi.org/10.1016/j.bpj.2019.04.032). Q-FADD uses simulated free diffusion models to fit experimental DNA microirradiation accumulation timeseries data, and the `qFADD.py` program utilizes a grid-search routine and multiple replicates per gridpoint to remove as much of the fitting procedure as possible from user bias.

This package comes with the files necessary to run qFADD.py in both command-line and GUI environments. Because `qFADD.py` makes use of the `mpi4py` module, installation of the code to high-performance parallel computing environments is possible. This repository also acts as the base code for the publicly-accessible Q-FADD webserver module (https://qfadd.colorado.edu).

Example use cases of the `qFADD.py` and the `image_analyzer.py` pre-processing program can be found in the [example_usage](https://github.com/Luger-Lab/Q-FADD/tree/master/example_usage) folder. These examples are meant to explain the basic workflow of conducting a Q-FADD analysis, and modifications to certain parameter values may be required to appropriately model your system of interest.

If you use `qFADD.py` in your research, please cite both the original article (https://doi.org/10.1016/j.bpj.2019.04.032) and the follow-up `qFADD.py` manuscript (article upcoming).
