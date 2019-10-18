#!/bin/bash

image_analyzer.py -m_file 1.31.18_GFPP1_Hela_1min_003.nd2 -roi_file 1.31.18_GFPP1_Hela_1min_003_ROI.tif -o precalc-single_file_example/ -nchan 'DAPI' -pchan 'EGFP' -x_buff "-10" -y_buff "10" -n_rois 0 -pre_frames 6 --drift_correct --bg_correct

cd precalc-single_file_example/

mpirun -np 25 qFADD.py -mf 1.31.18_GFPP1_Hela_1min_003NuclMask.txt -roif 1.31.18_GFPP1_Hela_1min_003ROI.txt -kf 1.31.18_GFPP1_Hela_1min_003_normalized.csv -out single_file_example -t0 12.0 -roic 2 -norm 6 -nmol 10000 -mobile_min 100 -mobile_max 1000 -mobile_step 100 -Dmin 1 -Dmax 15 -Dstep 1 -pix 0.08677 -tstep 0.1824 -nsteps 330 -ensemble_size 3 &> single_file_example_qfadd.log
