#!/bin/bash

image_analyzer.py -batch_dir ./ -m_ext ".nd2" -roi_ext "_ROI.tif" -o precalc-batch_directory_example/ -nchan 'DAPI' -pchan 'EGFP' -x_buff "-10" -y_buff "10" -n_rois 0 -pre_frames 6 --drift_correct --bg_correct

