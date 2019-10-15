# qFADD.py Installation Instructions

This document details the steps required to install the qFADD.py program.
This procedure will install both the command-line and GUI versions of the programs.

1. qFADD.py requires prerequisite Python and Java packages, as well as a functioning MPI environment (through
libraries such as OpenMPI).
    
    a. For Python, install the open-source Anaconda (v3) distribution, in accordance to your computing environment
and Anaconda's installation instructions. Installations have been successfully tested on CentOS 7 machines, but most Linux environments should be supported.
    
    b. A functioning Java environment is required to run the `python-bioformats` package (relies on `javabridge` module).
    Excessive version testing has not been conducted, but our installations were successful with the Java Development Kit (v11.0.2).

2. Use `conda` and `pip` to install the necessary packages for running qFADD-related programs:
    
    a. `conda install ffmpeg pyside2`
    
    b. `pip install msgpack python-bioformats mpi4py nd2reader==3.1.0 matplotlib-scalebar opencv-python`

3. Download the qFADD code from this repository: 
