#!/usr/bin/env python

from MovieAnalyzer_library import *
import javabridge
import bioformats
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m_file",dest='m_file',help='Input movie file for single movie processing.',type=str,default='')
parser.add_argument("-roi_file",dest='roi_file',help='Input .tiff file for capturing ROI',type=str,default='')
parser.add_argument("-batch_dir",dest='in_dir',help='Input directory containing the filelist for batch processing.',type=str,default='')
parser.add_argument("-m_ext",dest='m_ext',help='Extension for nuclear movies within \"-batch_dir\" (Default = ".nd2")',type=str,default='.nd2')
parser.add_argument("-roi_ext",dest='roi_ext',help='Extension for ROI definition files within \"-batch_dir\" (Default = "_ROI.tif")',type=str,default='_ROI.tif')
parser.add_argument("-o",dest='out_dir',help='Output directory for saving processed file information. (Default = "./")',type=str,default="./")
parser.add_argument("-nchan",dest='nuc_chan',help='Name of channel used to track the nuclear envelope. (Default = "DAPI")',type=str,default="DAPI")
parser.add_argument("-pchan",dest='prot_chan',help='Name of channel used to track protein accumulation. (Default = "EGFP")',type=str,default="EGFP")
parser.add_argument("-x_buff",dest='x_buff',help='Number of pixels to buff (or reduce, if negative) X-dimension of the ROI to correct for ROI border smearing artifacts',type=int,default=0)
parser.add_argument("-y_buff",dest='y_buff',help='Number of pixels to buff (or reduce, if negative) Y-dimension of the ROI to correct for ROI border smearing artifacts',type=int,default=0)
parser.add_argument("-n_rois",dest='n_roi',help='Number of additional  ROIs to include in both (vertical) directions from the irradiation ROI. (Default = 0)',type=int,default=0)
parser.add_argument("-pre_frames",dest='pre_frames',help='Number of frames in movie prior to irradiation. (Default = 0)',type=int,default=0)
parser.add_argument("--drift_correct",dest='drift_correct',action='store_true',help='Correct movie stacks for nuclear drift')
parser.add_argument("--bg_correct",dest='bg_correct',action='store_true',help='Correct for bleaching effects')
args = parser.parse_args()


t0 = time.time()
analyzer = RecruitmentMovieAnalyzer()

#Start the javabridge VM before you do the running
javabridge.start_vm(class_path=bioformats.JARS)

#Input Parsing etc.
if args.m_file != '':
    if args.roi_file == '':
        print("\n\n\tERROR: An associated \"-roi_file\" is required when declaring a \"-m_file\" for processing!!!\n\n")
        javabridge.kill_vm()
        quit()
    analyzer.LoadFile(video_file=args.m_file,roi_file=args.roi_file)
elif args.in_dir != '':
    analyzer.LoadDirectory(video_direct=args.in_dir,extension=args.m_ext,roi_extension=args.roi_ext)
else:
    print("\n\n\tERROR: Either a movie file (-m_file) or a directory of movie files (-batch_dir) must be supplied!!!\n\n")
    javabridge.kill_vm()
    quit()
analyzer.SetParameters(protein_channel=args.prot_chan,nuclear_channel=args.nuc_chan,additional_rois=args.n_roi,bleach_frames=args.pre_frames,save_direct=args.out_dir,roi_buffer=[args.x_buff,args.y_buff],irrad_frame=args.pre_frames,bg_correct=args.bg_correct,track_nucleus=args.drift_correct)

#Image processing
analyzer.ProcessFileList()
analyzer.ClearFileList()
javabridge.kill_vm()

#Timing Report
tf = time.time()
print("Processing complete in "+str((tf-t0)/60.)+" minutes")
