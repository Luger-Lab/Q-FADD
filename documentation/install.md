# qFADD.py Installation Instructions

This document details the steps required to install the qFADD.py program.
This procedure will install both the command-line and GUI versions of the programs.

1. qFADD.py requires prerequisite Python and Java packages, as well as a functioning MPI environment (through
libraries such as OpenMPI).
    
    a. For Python, install the open-source Anaconda (v3) distribution, in accordance to your computing environment
and Anaconda's installation instructions. Installations have been successfully tested on CentOS 7 machines, but most Linux environments should be supported (although have not been thoroughly tested).
    
    b. A functioning Java environment is required to run the `python-bioformats` package (relies on `javabridge` module).
    Excessive version testing has not been conducted, but our installations were successful with the Java Development Kit (v11.0.2).

2. Use `conda` and `pip` to install the necessary packages for running qFADD-related programs:
    
    a. `conda install ffmpeg pyside2`
    
    b. `pip install msgpack python-bioformats mpi4py nd2reader==3.1.0 matplotlib-scalebar opencv-python`

3. Download the qFADD code from this repository: `git clone github.com/sbowerma/qFADD.py`

4. Enter the `src/` directory, and run the `install.py` script.

    a. For the `-install_path` option, you should either direct the `install.py` to install the programs to a folder within your 
    `PATH` variable (such as the `bin/` folder of your Anaconda installation), or else you should manually add the `-install_path` 
    location to your `PATH` variable.
    
5. You're done!
    
    a. You can use the files in the [example_usage](https://github.com/sbowerma/qFADD.py/tree/master/example_usage) directory to test the stability of your installation.
