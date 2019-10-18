#!/usr/bin/env python
import argparse
from qFADD_library import *
import mpi4py
mpi4py.rc.recv_mprobe = False #Bugfix recommended online
from mpi4py import MPI

##### MPI ENVIRONMENT #####
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
###########################


##################### Command Line Options  ###########################################################
parser  = argparse.ArgumentParser()
parser.add_argument("-mf",dest='mask_file',help='Mask File',type=str,required=True)
parser.add_argument("-roif",dest='roi_file',help='ROI File',type=str,required=True)
parser.add_argument("-kf",dest='kinetics_file',help='Kinetics File',type=str,required=True)
parser.add_argument("-out",dest='output_prefix',help='Prefix for output files',type=str,default='qFADD_output')
parser.add_argument("-t0",dest='t0',help='Offset time to line up experiment and model',type=float,default=12)
#parser.add_argument("-cstart",dest='column_start',help='First column to pull ROI intensity data from',type=int,default=8)
#parser.add_argument("-cend",dest='column_end',help='Last column to pull ROI intensity data from',type=int,default=14)
parser.add_argument("-roic",dest='roi0_column',help='Column of "-kf" file containing data for the roi0 region',type=int,default=11)
parser.add_argument("-norm",dest='norm_frames',help='Number of frames to use for intensity normalization',type=int,default=6)
parser.add_argument("-nmol",dest='particles',help='Number of molecules',type=int,default=1000)
#parser.add_argument("-mobile",dest='mobile_part',help='Mobile fraction of particles (as molecules per 1000). (Default = 600)',type=int,default=600)
parser.add_argument("-mobile_min",dest='mobile_min',help='Minimum mobile fraction of particles for grid search. (Default = 600)',type=int,default=600)
parser.add_argument("-mobile_max",dest='mobile_max',help='Maximum mobile fraction of particles for grid search. (Default = 1000)',type=int,default=1000)
parser.add_argument("-mobile_step",dest='mobile_step',help='Step size in mobile fraction for grid search. (Default = 100)',type=int,default=100)
parser.add_argument("-pix",dest='scale',help='Pixel size for movie images (in micro-meters)',type=float,default=0.08677)
parser.add_argument("-tstep",dest='tstep',help='Timestep (in seconds)',default=0.1824,type=float)
parser.add_argument("-nsteps",dest='nsteps',help='Number of Monte Carlo iterations',type=int,default=350)
#parser.add_argument("-D",dest='D',help='Set a model value for the diffusion coefficient (in units of pixels per frame)',type=int,default=0)
parser.add_argument("-Dmin",dest='Dmin',help='Minimum value for D (in units of pixels per frame) for automated search',type=int,default=6)
parser.add_argument("-Dmax",dest='Dmax',help='Maximum value for D (in units of pixels per frame) for automated search',type=int,default=6)
parser.add_argument("-Dstep",dest='Dstep',help='Step size in D for grid search.  (Default = 1)',type=int,default=1)
parser.add_argument("-circle_radius",dest='roi_radius',help='Radius of theoretical ROI0 circle',type=int,default=0)
parser.add_argument("-ensemble_size",dest='ensemble_size',help='Number of Diffusion Models to run',type=int,default=1)
parser.add_argument("--find_plat",dest='find_plat',action='store_true',help='Determine simulation time from the point in which the experimental profile plateaus')
parser.add_argument("--use_mobile_formula",dest='use_f_formula',action='store_true',help='Only sample mobile fraction values around a pre-calculated approximate value')
parser.add_argument("--save_movie",dest='save_movie',action='store_true',help='Save a .mp4 file containing the simulated particle trajectories. Warning: will increase simulation time!')
parser.add_argument("--use_avg",dest='use_avg',action='store_true',help='Report the average of sampled trajectories instead of the median sampled trajectory as the quality of fit')
parser.add_argument("--plot_all",dest='plot_all',action='store_true',help='Plot the model intensity timeseries vs experimental for every grid point sampled, not only for the best-fit model.')
args = parser.parse_args()
################################################################################################################

diffusion_simulator = NuclearDiffusion()
diffusion_simulator.main(args)
