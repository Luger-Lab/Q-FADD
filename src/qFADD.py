#!/usr/bin/env python
import argparse
from qFADD_library import *
import mpi4py
mpy4py.rc.recv_mprobe
from mpi4py import MPI

####### MPI ENVIRONMENT #####
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#############################

############### Command Line Options ###############################
parser  = argparse.ArgumentParser()
parser.add_argument("-mf",dest='mask_file',help='Mask File',type=str,required=True)
parser.add_argument("-roif",dest='roi_file',help='ROI File',type=str,required=True)
parser.add_argument("-kf",dest='kinetics_file',help='Kinetics File',type=str,required=True)
parser.add_argument("-out",dest='output_prefix',help='Prefix for output files',type=str,default='qFADD_output')
parser.add_argument("-t0",dest='t0',help='Offset time to line up experiment and model',type=float,default=12)
parser.add_argument("-cstart",dest='column_start',help='First column to pull ROI intensity data from',type=int,default=8)
parser.add_argument("-cend",dest='column_end',help='Last column to pull ROI intensity data from',type=int,default=14)
parser.add_argument("-norm",dest='norm_frames',help='Number of frames to use for intensity normalization',type=int,default=6)
parser.add_argument("-nmol",dest='particles',help='Number of molecules',type=int,default=1000)
parser.add_argument("-mobile_min",dest='mobile_min',help='Minimum mobile fraction of particles for grid search. (Default=600)',type=int,default=600)
parser.add_argument("-mobile_max",dest='mobile_max',help='Maximum mobile fraction of particles for grid search. (Default=1000)',type=int,default=1000)
parser.add_argument("-mobile_step",dest='mobile_step',help='Step size in mobile fraction for building the grid search. (Default=100)',type=int,default=100)
parser.add_argument("-Dmin",dest='Dmin',help='Minimum value for D (in units of pixels per frame) for grid search. (Default=4)',type=int,default=4)
parser.add_argument("-Dmax",dest='Dmax',help='Maximum value for D (in units of pixels per frame) for grid search. (Default=6)',type=int,default=6)
parser.add_argument("-Dstep",dest='Dstep',help='Step size in D for building the grid search. (Default = 1)',type=int,default=1)
parser.add_argument("-ensemble_size",dest='ensemble_size',help='Number of replicates to make per grid point. (Default = 1)',type=int,default=1)
parser.add_argument("--find_plat",dest='find_plat',action='store_true',help='Determine simulation time from the point in which the experimental profile plateaus')
parser.add_argument("--save_movie",dest='save_movie',action='store_true',help='Save a .mp4 file containing the simulated particle trajectories.  Warning: will significantly increase simulation time!')
args = parser.parse_args()
####################################################################

diffusion_simulator = NuclearDiffusion()
diffusion_simulater.main(args)
