#!/usr/bin/env python
import os,time,datetime
import glob
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.animation as animate
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import nd2reader as nd2
import bioformats,javabridge
import warnings
warnings.filterwarnings("ignore")
from skimage import feature,morphology,restoration #Edge detection
from skimage.transform import warp,SimilarityTransform
from skimage.feature import ORB, match_descriptors,register_translation
from skimage import measure
from skimage.measure import ransac
from skimage.filters import sobel
from skimage.color import label2rgb
import scipy,cv2
from scipy import ndimage as ndi
from scipy.signal import savgol_filter

from sklearn.cluster import DBSCAN
from sklearn import metrics

class RecruitmentMovieAnalyzer(object):
    def __init__(self):
        self.nuclear_channel= 'TRITC'
        self.irrad_frame    = 'auto'
        self.roi_buffer     = [-10,0]
        self.track_nucleus  = True
        self.autosave       = True
        self.save_direct    = './MovieAnalysis_output/'
        self.save_movie     = True
        self.save_roi_data  = True
        self.additional_rois= 0
        #self.correct_bleach = True
        self.bleach_frames  = 0
        self.threshold      = -1 
        self.bg_correct     = True
        self.bleach_correct = True

    def SetParameters(self,nuclear_channel='TRITC',protein_channel='EGFP',irrad_frame='auto',roi_buffer=[-10,0],track_nucleus=True,autosave=True,save_direct='./MovieAnalysis_output/',save_movie=True,save_roi_data=True,additional_rois=0,bleach_correct=True,bleach_frames=0,threshold=-1,bg_correct=True,verbose=True):
        self.nuclear_channel= nuclear_channel
        self.protein_channel= protein_channel
        self.irrad_frame    = irrad_frame
        self.roi_buffer     = roi_buffer
        self.track_nucleus  = track_nucleus
        self.autosave       = autosave
        self.save_direct    = save_direct
        self.save_movie     = save_movie
        self.save_roi_data  = save_roi_data
        self.additional_rois= additional_rois
        #self.correct_bleach = correct_bleach
        self.bleach_frames  = bleach_frames
        self.threshold      = threshold
        if str(self.threshold).lower() == 'auto':
            self.threshold  = 3
        self.bg_correct     = bg_correct
        self.bleach_correct = bleach_correct
        self.verbose        = verbose

        if not os.path.isdir(self.save_direct):
            os.mkdir(self.save_direct)
        else:
            print("WARNING: Directory "+self.save_direct+" already exists!  Be aware that you may be overwriting files!!!")

#        if self.save_movie:
#            self.ffmpeg_writer   = animate.writers['ffmpeg']

    def LoadFile(self,video_file='',roi_file=''):
        if not os.path.isfile(video_file):
            print("ERROR: Cannot load file - "+video_file+" - File not found!")
            vid_exists = False
        else:
            vid_exists = True

        if not os.path.isfile(roi_file):
            print("ERROR: Cannot load file - "+roi_file+" - File not found!")
            roi_exists = False
        else:
            roi_exists = True
        
        if roi_exists and vid_exists:
            try:
                self.video_list.append(video_file)
            except:
                self.video_list = [video_file]
            try:
                self.roif_list.append(roi_file)
            except:
                self.roif_list  = [roi_file]
        else:
            print("File(s) missing.  Cannot load desired experiment for analysis!!!")

    def LoadDirectory(self,video_direct='',extension='.nd2',roi_extension='_ROI.tif'):
        if not os.path.isdir(video_direct):
            print("ERROR: Cannot load directory - "+video_direct+" - Directory not found!")
        else:
            self.video_direct   = video_direct
            filelist            = glob.glob(os.path.join(video_direct,"*"+extension))
            for vidFile in filelist:
                roiFile = vidFile.replace(extension,roi_extension)
                if not os.path.isfile(roiFile):
                    print("WARNING: Could not find ROI file ("+roiFile+") for video file ("+vidFile+")! Not adding files to processing list!!!")
                else:
                    try:
                        self.video_list.append(vidFile)
                    except:
                        self.video_list = [vidFile]
                    try:
                        self.roif_list.append(roiFile)
                    except:
                        self.roif_list  = [roiFile]


    def ClearFileList(self):
        self.video_list = []
        self.Nfiles = 0

    def ProcessOther(self,input_video):
        #Process Metadata
        this_omexmlstr = bioformats.get_omexml_metadata(input_video)
        this_omexml = bioformats.OMEXML(this_omexmlstr)
        these_pixels= this_omexml.image().Pixels
        this_numt   = these_pixels.get_SizeT()
        this_numc   = these_pixels.get_SizeC()
        self.pix_res= these_pixels.get_PhysicalSizeX()
        try:
            final_time      = these_pixels.Plane(index=this_numt*this_numc-1).DeltaT
            self.has_time   = True
            if os.path.splitext(input_video)[1]==".czi":
                #Zeiss microscopes don't count from zero, so need to correct for first time point
                self.init_time  = these_pixels.Plane(index=0).DeltaT
                self.is_zeiss   = True
                final_time = final_time - self.init_time
            else:
                self.is_zeiss   = False
        except:
            self.has_time   = False
            print("Warning: Unable to extract time points from movie! Please extract them by hand!")

        print("Loading \""+input_video+"\" :")
        if self.has_time:
            print("\t\tMovie Length     = "+str(np.around(final_time,decimals=2))+" seconds.")
        else:
            print("\t\tMovie Length     = "+str(this_numt)+" frames.")
        print("\t\tPixel Resolution = "+str(self.pix_res)+" um")

        this_tsteps = np.zeros(this_numt,dtype=float) # We have to fill this up as we open images in bioformats...
        self.roi_intensity_array    = np.zeros((len(this_tsteps),1+(1+2*self.additional_rois)),dtype=int)
        self.roi0_intensity_array   = np.zeros((len(this_tsteps),2),dtype=int)
        self.total_intensity_array  = np.zeros(len(this_tsteps),dtype=float)


        for self.ts in range(this_numt):
            if self.has_time:
                this_tsteps[self.ts] = these_pixels.Plane(index=this_numc*self.ts).DeltaT
                if self.is_zeiss:
                    this_tsteps[self.ts] -= self.init_time
            else:
                this_tsteps[self.ts] = self.ts
            this_nuc_frame =  np.array(bioformats.load_image(input_video,c=self.nuclear_channel,t=self.ts,rescale=False))
            this_nuc_path,this_nuc_points,this_nuc_fill = self.getNuclearMask(this_nuc_frame,sigma=self.threshold)
            if self.ts==0:
                self.first_nuc_points= np.copy(this_nuc_points)
                self.first_nuc_frame = np.copy(this_nuc_frame)
                self.first_nuc_fill  = np.copy(this_nuc_fill)
                shifted_nuc_fill     = np.copy(this_nuc_fill)
            else:
                if self.track_nucleus:
                    shift,error,diffphase= register_translation(self.first_nuc_fill,this_nuc_fill)
                else:
                    shift = np.array([0,0],dtype=int)

                if (shift[0]!=0) or (shift[1]!=0):
                    shifted_nuc_fill = np.zeros_like(this_nuc_fill)
                    shifted_nuc_frame= np.zeros_like(this_nuc_frame)
                    N1 = len(shifted_nuc_fill)
                    N2 = len(shifted_nuc_fill[0])
                    for idx in range(N1):
                        for idx2 in range(N2):
                            if (idx - shift[0] >= 0) and (idx2 - shift[1] >= 0) and (idx - shift[0] < N1) and (idx2-shift[1] < N2):
                                shifted_nuc_fill[idx,idx2] = this_nuc_fill[idx-int(shift[0]),idx2-int(shift[1])]
                    this_nuc_points[:,0] -= int(shift[0])
                    this_nuc_points[:,1] -= int(shift[1])
                else:
                    shifted_nuc_fill = np.copy(this_nuc_fill)

            this_prot_frame = np.array(bioformats.load_image(input_video,c=self.protein_channel,t=self.ts,rescale=False))
            if self.bg_correct:
                this_chan_frame = self.correctBackground(this_prot_frame,input_video,self.protein_channel)
            else:
                this_chan_frame = np.copy(this_prot_frame)

            
            this_vmin = np.min(this_prot_frame)
            if self.ts == 0:
                shifted_frame = np.copy(this_chan_frame)
            else:
                if (shift[0]!=0) or (shift[1]!=0):
                    shifted_frame = np.zeros_like(this_prot_frame)
                    N1 = len(shifted_frame)
                    N2 = len(shifted_frame[0])
                    for idx in range(N1):
                        for idx2 in range(N2):
                            if (idx-shift[0] >= 0) and (idx2-shift[1] >= 0) and (idx-shift[0] < N1) and (idx2-shift[1] < N2):
                                shifted_frame[idx,idx2] = this_chan_frame[idx-int(shift[0]),idx2-int(shift[1])]
                else:
                    shifted_frame = np.copy(this_chan_frame)

            if self.save_movie:
                self.SaveMovieFrame(this_chan_frame,roi_path_dict=self.roi_path_dict,vmin=this_vmin,suffix='raw')
                self.SaveMovieFrame(shifted_frame,roi_path_dict=self.roi_path_dict,vmin=this_vmin,suffix='shifted')

            for idx in range(-self.additional_rois,self.additional_rois+1):
               this_roi_buffed  = self.roi_buff_dict[idx]
               this_roi_path    = self.roi_path_dict[idx]
               this_roi_pix     = np.where(np.logical_and(this_roi_buffed>0,shifted_nuc_fill>0))
               roi_intensity    = np.sum(shifted_frame[this_roi_pix])
               this_col_id      = idx + self.additional_rois + 1
               self.roi_intensity_array[self.ts,this_col_id] = roi_intensity
               if idx == 0:
                   self.roi0_intensity_array[self.ts,1] = roi_intensity

            all_nuc_prot_pix = np.where(shifted_nuc_fill > 0)
            total_intensity  = np.sum(shifted_frame[all_nuc_prot_pix])
            self.total_intensity_array[self.ts] = total_intensity

        if self.bleach_frames > 0 and self.bg_correct:
            pre_bleached = np.average(self.total_intensity_array[:self.bleach_frames])
            self.bleach_corrections = np.divide(pre_bleached,self.total_intensity_array)
            self.roi_intensity_array[:,1:] = self.roi_intensity_array[:,1:]\
                                          * self.bleach_corrections[:,np.newaxis]
            self.bleach_correct_tot = np.multiply(self.total_intensity_array,self.bleach_corrections)

            self.normalized_intensity_array = np.copy(self.roi_intensity_array).astype(float)
            for colID in range(len(self.roi_intensity_array[0,1:])):
                this_col = self.normalized_intensity_array[:,1+colID].astype(float)
                self.normalized_intensity_array[:,1+colID] = this_col/np.average(this_col[:self.bleach_frames])
        else:
            self.bleach_correct_tot = np.copy(self.total_intensity_array)
            self.normalized_intensity_array = np.copy(self.roi_intensity_array).astype(float)
            for colID in range(len(self.roi_intensity[:,1:])):
                this_col = self.normalized_intensity_array[:,1+colID].astype(float)
                self.normalized_intensity_array[:,1+colID] = this_col/this_col[0]

        #Print the intensity timeseries
        ofile   = open(self.this_ofile,'w')
        nofile  = open(self.this_nofile,'w')
        ofile.write('Input Filename: '+input_video+'\n')
        nofile.write('Input Filename: '+input_video+'\n')
        now = datetime.datetime.now()
        ofile.write('Analysis Date: '\
                   +now.strftime('%d-%m-%Y %H:%M:%S')\
                   +'\n')
        nofile.write('Analysis Date: '\
                    +now.strftime('%d-%m-%Y %H:%M:%S')\
                    +'\n')

        N_columns = 2 * self.additional_rois + 1 #All the ROIs, including 0
        N_columns += 1 #Account for the time column
        chan_center= 1+self.additional_rois
        for idx in range(N_columns):
            if idx==chan_center:
                ofile.write(self.protein_channel)
                nofile.write(self.protein_channel)
            ofile.write(',')
            nofile.write(',')
        ofile.write("\nTime (s)")
        nofile.write("\nTime (s)")
        roi_tracker = np.arange(-self.additional_rois,self.additional_rois+1)
        for idx in range(N_columns-1):
            ofile.write(",ROI "+str(roi_tracker[idx]))
            nofile.write(",ROI "+str(roi_tracker[idx]))
        ofile.write('\n')
        nofile.write('\n')
        for tidx in range(this_numt):
            ofile.write(str(this_tsteps[tidx]/1000.))
            nofile.write(str(this_tsteps[tidx]/1000.))
            for cidx in range(1,N_columns):
                ofile.write(","+str(self.roi_intensity_array[tidx,cidx]))
                nofile.write(","+str(self.normalized_intensity_array[tidx,cidx]))
            ofile.write("\n")
            nofile.write("\n")
        ofile.close()
        nofile.close()

        #Make the intensity plot
        plt.figure(figsize=(6,4))
        plot_array  = np.genfromtxt(self.this_nofile,skip_header=4,delimiter=',')
        for idx in range(self.additional_rois+1):
            plt.plot(plot_array[:,0],plot_array[:,idx+1],linestyle='',
                     marker='.',markersize=5,label='ROI '+str(roi_tracker[idx]))
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Intensity (A.U.)")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(self.this_noplot,format='pdf')

        if self.save_movie:
            movie_basename  = os.path.basename(input_video)
            extension       = "."+movie_basename.split(".")[1]
            raw_movie_file  = os.path.join(self.save_direct,
                                          movie_basename.replace(
                                          extension,'_raw.mp4'))
            shifted_movie_file = os.path.join(self.save_direct,
                                          movie_basename.replace(
                                          extension,"_drift_corrected.mp4"))
            if os.path.isfile(raw_movie_file):
                os.remove(raw_movie_file)
            if os.path.isfile(shifted_movie_file):
                os.remove(shifted_movie_file)

            print(shifted_movie_file)

            os.system("ffmpeg -r 30 -f image2 -i "+os.path.join(self.save_direct,
                      "frame%04d_raw.png")+" -vcodec libx264"\
                      +" -crf 25 -pix_fmt yuv420p "\
                      +raw_movie_file+" &> raw_movie_ffmpeg.log")
            os.system("ffmpeg -r 30 -f image2 -i "+os.path.join(self.save_direct,
                      "frame%04d_shifted.png")+" -vcodec libx264"\
                      +" -crf 25 -pix_fmt yuv420p "\
                      +shifted_movie_file+" &> drift_corrected_movie_ffmpeg.log")

            movie_frames = glob.glob(os.path.join(self.save_direct,"*.png"))
            for FRAME in movie_frames:
                os.remove(FRAME)

    def ProcessND2(self,input_video):
        this_video  = nd2.reader.ND2Reader(input_video)
        this_tsteps = this_video.get_timesteps()
        this_chans  = np.array(this_video.metadata['channels'],dtype=str)
        this_pix    = this_video.metadata['pixel_microns']
        self.pix_res= float(this_pix)
        
        print("Loading \""+input_video+"\" :")
        print("\t\tMovie length     = "+str(np.around(this_tsteps[-1]/1000.,decimals=2))+" seconds.")
        print("\t\tChannel Names    = "+str(this_chans))
        print("\t\tPixel Resolution = "+str(this_pix)+" um")

        nuc_chan_check  = np.where(this_chans==self.nuclear_channel)[0]

        if len(nuc_chan_check)==0:
            print("ERROR: Nuclear channel( \""+self.nuclear_channel+"\") not found!! Channel List = "+str(this_chans))
            print("ERROR: File (\""+input_video+"\") not processed!!!")
            return -1
        elif len(nuc_chan_check)>1:
            print("ERROR: Nuclear channel (\""+self.nuclear_channel+"\") is not unique!! Channel List = "+str(this_chans))
            print("ERROR: File (\""+input_video+"\") not processed!!!")
            return -1
        else:
            nuc_chan    = nuc_chan_check[0]

        prot_chan_check = np.where(this_chans==self.protein_channel)[0]
        if len(prot_chan_check) == 0:
            print("ERROR: Protein channel (\""+self.protein_channel+"\") not found!! Channel List = "+str(this_chans))
            print("ERROR: File (\""+input_video+"\") not processed!!!")
            return -1
        elif len(prot_chan_check) > 1:
            print("ERROR: Protein channel (\""+self.protein_channel+"\") is not unique!! Channel List = "+str(this_chans))
            print("ERROR: File (\""+input_video+"\") not processed!!!")
            return -1
        else:
            prot_chan   = prot_chan_check[0]

        #Build the intensity timeseries array
        self.roi_intensity_array    = np.zeros((len(this_tsteps),1+(1+2*self.additional_rois)),dtype=int)
        self.roi0_intensity_array   = np.zeros((len(this_tsteps),2),dtype=int)
        self.total_intensity_array  = np.zeros(len(this_tsteps),dtype=float)

        for self.ts in range(len(this_video)):
            this_nuc_frame  = this_video.get_frame_2D(c=nuc_chan,t=self.ts)
            this_nuc_path,this_nuc_points,this_nuc_fill = self.getNuclearMask(this_nuc_frame,sigma=self.threshold)

            if self.ts==0:
                self.first_nuc_points= np.copy(this_nuc_points)
                self.first_nuc_frame = np.copy(this_nuc_frame)
                self.first_nuc_fill  = np.copy(this_nuc_fill)
                shifted_nuc_fill= np.copy(this_nuc_fill)
            else:
                if self.track_nucleus:
                    shift,error,diffphase= register_translation(self.first_nuc_fill,this_nuc_fill)
                else:
                    shift = np.array([0,0],dtype=int)
                if (shift[0]!=0) or (shift[1]!=0):
                    shifted_nuc_fill = np.zeros_like(this_nuc_fill)
                    shifted_nuc_frame= np.zeros_like(this_nuc_frame)
                    N1 = len(shifted_nuc_fill)
                    N2 = len(shifted_nuc_fill[0])
                    for idx in range(N1):
                        for idx2 in range(N2):
                            if (idx - shift[0] >= 0) and (idx2 - shift[1] >= 0) and (idx - shift[0] < N1) and (idx2-shift[1] < N2):
                                shifted_nuc_fill[idx,idx2] = this_nuc_fill[idx-int(shift[0]),idx2-int(shift[1])]
                    this_nuc_points[:,0] -= int(shift[0])
                    this_nuc_points[:,1] -= int(shift[1])
                else:
                    shifted_nuc_fill = np.copy(this_nuc_fill)

            this_prot_frame = this_video.get_frame_2D(c=prot_chan,t=self.ts)
            if self.bg_correct:
                this_chan_frame = self.correctBackground(this_prot_frame,this_video,prot_chan,is_nd2=True)
            else:
                this_chan_frame = np.copy(this_prot_frame)
            #Need to know minimium pixel count to adjust movie brightness
            this_vmin = np.min(this_prot_frame)
            if self.ts == 0:
                shifted_frame = np.copy(this_prot_frame)
            else:
                if (shift[0]!=0) or (shift[1]!=0):
                    shifted_frame = np.zeros_like(this_prot_frame)
                    N1 = len(shifted_frame)
                    N2 = len(shifted_frame[0])
                    for idx in range(N1):
                        for idx2 in range(N2):
                            if (idx-shift[0] >= 0) and (idx2-shift[1] >= 0) and (idx - shift[0] < N1) and (idx2-shift[1] < N2):
                                shifted_frame[idx,idx2] = this_chan_frame[idx-int(shift[0]),idx2-int(shift[1])]
                else:
                    shifted_frame = np.copy(this_chan_frame)
                    
            if self.save_movie:
                self.SaveMovieFrame(this_chan_frame,roi_path_dict=self.roi_path_dict,vmin=this_vmin,suffix='raw')
                self.SaveMovieFrame(shifted_frame,roi_path_dict=self.roi_path_dict,vmin=this_vmin,suffix='shifted')

            for idx in range(-self.additional_rois,self.additional_rois+1):
                this_roi_buffed = self.roi_buff_dict[idx]
                this_roi_path   = self.roi_path_dict[idx]
                this_roi_pix    = np.where(np.logical_and(this_roi_buffed>0,shifted_nuc_fill>0))

                roi_intensity   = np.sum(shifted_frame[this_roi_pix])
                this_col_id = idx + self.additional_rois + 1
                self.roi_intensity_array[self.ts,this_col_id]   = roi_intensity
                if idx==0:
                    self.roi0_intensity_array[self.ts,1] = roi_intensity

            #Determine total intensity for bleach correction
            all_nuc_prot_pix = np.where(shifted_nuc_fill > 0)
            total_intensity  = np.sum(shifted_frame[all_nuc_prot_pix])
            self.total_intensity_array[self.ts] = total_intensity

        if self.bleach_frames > 0 and self.bg_correct:
            pre_bleached = np.average(self.total_intensity_array[:self.bleach_frames])
            self.bleach_corrections = np.divide(pre_bleached,self.total_intensity_array)
            self.roi_intensity_array[:,1:] = self.roi_intensity_array[:,1:]\
                                           * self.bleach_corrections[:,np.newaxis]
            self.bleach_correct_tot = np.multiply(self.total_intensity_array,self.bleach_corrections)

            self.normalized_intensity_array= np.copy(self.roi_intensity_array).astype(float)
            for colID in range(len(self.roi_intensity_array[0,1:])):
                this_col = self.normalized_intensity_array[:,1+colID].astype(float)
                self.normalized_intensity_array[:,1+colID] = this_col/np.average(this_col[:self.bleach_frames])
        else:
            self.bleach_correct_tot = np.copy(self.total_intensity_array)
            self.normalized_intensity_array = np.copy(self.roi_intensity_array).astype(float)
            for colID in range(len(self.roi_intensity_array[:,1:])):
                this_col = self.normalized_intensity_array[:,1+colID].astype(float)
                self.normalized_intensity_array[:,1+colID] = this_col/this_col[0]
        
        #Pring the intensity timeseries
        ofile = open(self.this_ofile,'w')
        nofile= open(self.this_nofile,'w')
        ofile.write("Input Filename: "+input_video+"\n")
        nofile.write("Input Filename: "+input_video+"\n")
        now = datetime.datetime.now()
        ofile.write("Analysis Date: "\
                   +now.strftime("%d-%m-%Y %H:%M:%S")\
                   +"\n")
        nofile.write("Analysis Date: "\
                    +now.strftime("%d-%m-%Y %H:%M:%S")\
                    +"\n")

        N_columns = 2*self.additional_rois + 1 #All the ROIs, including 0
        N_columns += 1 #Account for the time idx
        chan_center = 1+self.additional_rois
        for idx in range(N_columns):
            if idx == chan_center:
                ofile.write(self.protein_channel)
                nofile.write(self.protein_channel)
            ofile.write(",")
            nofile.write(",")
        ofile.write("\nTime (s)")
        nofile.write("\nTime (s)")
        roi_tracker = np.arange(-self.additional_rois,self.additional_rois+1)
        for idx in range(N_columns-1):
            ofile.write(",ROI "+str(roi_tracker[idx]))
            nofile.write(",ROI "+str(roi_tracker[idx]))
        ofile.write("\n")
        nofile.write("\n")
        for tidx in range(len(this_tsteps)):
            ofile.write(str(this_tsteps[tidx]/1000.))
            nofile.write(str(this_tsteps[tidx]/1000.))
            for cidx in range(1,N_columns):
                ofile.write(","+str(self.roi_intensity_array[tidx,cidx]))
                nofile.write(","+str(self.normalized_intensity_array[tidx,cidx]))
            ofile.write("\n")
            nofile.write("\n")
        ofile.close()
        nofile.close()

        #Plot the intensities
        plt.figure(figsize=(6,4))
        plot_array  = np.genfromtxt(self.this_nofile,skip_header=4,
                                   delimiter=',')
        roi_idx = range(-self.additional_rois,self.additional_rois+1)
        for idx in range(self.additional_rois+1):
            plt.plot(plot_array[:,0],plot_array[:,idx+1],linestyle='',
                     marker='.',markersize=5,label='ROI '+str(roi_idx[idx]))
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Intensity (A.U.)")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(self.this_noplot,format='pdf')

        if self.save_movie:
            movie_basename  = os.path.basename(input_video)
            extension       = "."+movie_basename.split(".")[-1]
            raw_movie_file  = os.path.join(self.save_direct,
                                          movie_basename.replace(
                                          extension,'_raw.mp4'))
            shifted_movie_file = os.path.join(self.save_direct,
                                          movie_basename.replace(
                                          extension,"_drift_corrected.mp4"))
            if os.path.isfile(raw_movie_file):
                os.remove(raw_movie_file)
            if os.path.isfile(shifted_movie_file):
                os.remove(shifted_movie_file)

            os.system("ffmpeg -r 30 -f image2 -i "+os.path.join(self.save_direct,
                      "frame%04d_raw.png")+" -vcodec libx264"\
                      +" -crf 25 -pix_fmt yuv420p "\
                      +raw_movie_file+" &> raw_movie_ffmpeg.log")
            os.system("ffmpeg -r 30 -f image2 -i "+os.path.join(self.save_direct,
                      "frame%04d_shifted.png")+" -vcodec libx264"\
                      +" -crf 25 -pix_fmt yuv420p "\
                      +shifted_movie_file+" &> drift_corrected_movie_ffmpeg.log")

            movie_frames = glob.glob(os.path.join(self.save_direct,"*.png"))
            for FRAME in movie_frames:
                os.remove(FRAME)

    def SaveMovieFrame(self,frame,roi_path_dict={},vmin=0,suffix='raw'):
        if suffix=='raw':
            self.raw_frame_string = os.path.join(self.save_direct,
                                    "frame%04d"%(self.ts)+"_raw.png")
            save_string = self.raw_frame_string
        elif suffix=='shifted':
            self.shifted_frame_string = os.path.join(self.save_direct,
                                        "frame%04d"%(self.ts)+"_shifted.png")
            save_string = self.shifted_frame_string
        plt.figure(figsize=(6.,6.))
        ax = plt.subplot(111)
        ax.imshow(frame,vmin=vmin)
        ax.plot(self.first_nuc_points[:,1],self.first_nuc_points[:,0],color='red',linewidth=1.25)
        i=0
        for key in roi_path_dict:
            this_roi_path = roi_path_dict[key]
            this_patch = patches.PathPatch(this_roi_path,facecolor='none',linewidth=1.0,edgecolor='white')
            i+=1
            ax.add_patch(this_patch)

        scalebar = ScaleBar(self.pix_res,'um',location=4)
        plt.gca().add_artist(scalebar)
        plt.tight_layout()
        plt.savefig(save_string,format='png',dpi=300)
        
        plt.close('all')


    def ProcessFileList(self):
        self.Nfiles = len(self.video_list)
        for idx in range(self.Nfiles):
            input_video = self.video_list[idx]
            input_roif  = self.roif_list[idx]
            input_ext   = os.path.splitext(input_video)[1]

            self.output_prefix = os.path.join(self.save_direct,
                                 os.path.splitext(
                                 os.path.basename(input_video))[0])

            #File that contains the ROI coordinates
            this_roif   = np.array(bioformats.load_image(input_roif,rescale=False))
            self.roi_buff_dict,self.roi_path_dict = self.growROI(this_roif,self.roi_buffer)

            #Output file for intensity timeseries
            self.this_ofile = os.path.join(
                              self.save_direct,
                              os.path.basename(
                              input_video
                              ).replace(input_ext,'.csv')
                              )
            self.this_nofile = self.this_ofile.replace('.csv','_normalized.csv')
            self.this_noplot = self.this_nofile.replace('.csv','.pdf')
            #Currently tested importing nd2 files or TIFF files...theoretically can load any bioformats file
            if input_ext=='.nd2':
                self.ProcessND2(input_video)
            else:
                self.ProcessOther(input_video)

    def getNuclearMask2(self,nuclear_frame,show_plots=False,radius=20):
        temp_nuc= nuclear_frame.astype(float)
        if str(self.threshold).lower() =='auto':
            centerMask = np.zeros(np.shape(nuclear_frame),dtype=int)
            xval= np.arange(len(centerMask[0]))
            yval= np.arange(len(centerMask))
            #center the values around a "zero"
            xval= xval - np.median(xval)
            yval= yval - np.median(yval)        
     
            xval,yval = np.meshgrid(xvay,yval)

            #Determine threshold in a circle at the center of the frame
            #(assumes that you've centered your microscope on the cell)
            centerMask[np.where(xval**2+yval**2 < radius**2)] = 1

            #Calculate the mean and std intensity in the region
            mean_int= np.average(temp_nuc[centerMask])
            std_int = np.std(temp_nuc[centerMask])

            #Determine thresholding level
            self.thresh_level = mean_int - 0.5*mean_int
            #Check that the threshold level isn't too low
            if self.thresh_level <= 0:
                thresh_fact = 0.5
                while self.thresh_level < mean_int:
                    thresh_fact = thresh_fact - 0.1
                    self.thresh_level = (mean_int - thresh_fact * std_int)
        else:
            try:
                self.thresh_level = float(self.threshold)
                if np.isnan(self.thresh_level):
                    print("Could not understand setting for threshold ("\
                         +str(self.threshold_level)+"). Assuming \"auto\".")
                    self.threshold = 'auto'
                    self.getNuclearMask2(nuclear_frame,
                                         show_plots=show_plots,
                                         radius=radius)
            except:
                print("Could not understand setting for threshold ("\
                     +str(self.threshold_level)+"). Assuming \"auto\".")
                self.threshold = 'auto'
                self.getNuclearMask2(nuclear_frame,
                                     show_plots=show_plots,
                                     radius=radius)

        #Find all points in image above threshhold level
        thresh_masked = np.zeros(temp_nuc.shape,dtype=int)
        thresh_masked[np.where(temp_nuc>self.thresh_level)] = 1
        thresh_masked = self.imclearborderAnalogue(thresh_masked,8)
        thresh_masked = self.bwareaopenAnalogue(thresh_masked,500)
        thresh_masked = scipy.ndimage.binary_fill_holes(thresh_masked)
        

        labels = measure.label(thresh_masked,background=1)
        props  = measure.regionprops(labels)

        if len(np.unique(labels))>1:
            #We want the central object
            best_r = 99999.9
            
            xcent  = int(len(thresh_masked)/2)
            ycent  = int(len(thresh_masked[0])/2)

            for nuc_obj in props:
                this_center = nuc_obj.centroid
                this_r = np.sqrt((this_center[0] - xcent)**2\
                                +(this_center[1] - ycent)**2)
                if this_r < best_r:
                    best_r = this_r
                    center_nuc = nuc_obj
                    these_pix  = np.where(labels==nuc_obj.label)

        elif len(np.unique(labels))==1:
            these_pix = np.where(thresh_masked)

        else:
            print("ERROR: getNuclearMask2() could not find any nuclei! "\
                 +"Please specify a lower threshold.")
            quit()

        nuc_fill = np.zeros(np.shape(nuclear_frame),dtype=int)
        nuc_fill[these_pix] = 1.0
        nuc_fill = scipy.ndimage.binary_fill_holes(nuc_fill)

        for idx in range(len(nuclear_frame)):
            this_slice = np.where(nuc_fill[idx]>0)
            if len(this_slice[0]) > 0:
                this_min = this_slice[0][0]
                this_max = this_slice[0][-1]
                try:
                    nuc_points = np.vstack((nuc_points,[idx,this_min]))
                except:
                    nuc_points = np.array([idx,this_min])
                if this_max != this_min:
                    try:
                        nuc_points = np.vstack((nuc_points,[idx,this_max]))
                    except:
                        nuc_points = np.array([idx,this_max])

        nuc_points = np.vstack((nuc_points,nuc_points[0]))
        #Filter out the sharp edges
        nuc_points[:,1] = savgol_filter(nuc_points[:,1],51,3)
        nuc_path = Path(nuc_points,closed=True)

        if self.ts==0:
            self.saveNuclearMask(nuc_points)

        return nuc_path, nuc_points, nuc_fill


    def imclearborderAnalogue(self,image_frame,radius):
        #Contour the image
        Nx = len(image_frame)
        Ny = len(image_frame[0])
        img = cv2.resize(image_frame,(Nx,Ny))
        contours, hierarchy = cv2.findContours(img, cv2.RETR_FLOODFILL,
                              cv2.CHAIN_APPROX_SIMPLE)
        #Get dimensions
        nrows = image_frame.shape[0]
        ncols = image_frame.shape[1]

        #Track contours touching the border
        contourList = []

        for idx in range(len(contours)):
            this_contour = contours[idx]
            for point in this_contour:
                contour_row = point[0][1]
                contour_col = point[0][0]
                
                #Check if within radius of border, else remove
                rowcheck = (contour_row >= 0 and contour_row < radius)\
                           or (contour_row >= nrows-1-radius and contour_row\
                           < nrows)

                colcheck = (contour_col >= 0 and contour_col < radius)\
                           or (contour_col >= ncols-1-radius and contour_col\
                           < ncols)

                if rowcheck or colcheck:
                    contourList.append(idx)

        output_frame = image_frame.copy()
        for idx in contourList:
            cv2.drawContours(output_frame, contours, idx, (0,0,0), -1)

        return output_frame

    def bwareaopenAnalogue(self,image_frame,areaPix):
        output_frame = image_frame.copy()
        #First, identify all the contours
        contours,hierarchy = cv2.findContours(output_frame.copy(),
                                              cv2.RETR_FLOODFILL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        #then determine occupying area of each contour
        for idx in range(len(contours)):
            area = cv2.contourArea(contours[idx])
            if (area >= 0 and area <= areaPix):
                cv2.drawContours(output_frame,contours,idx,(0,0,0),-1)

        return output_frame

    def getNuclearMask(self,nuclear_frame,sigma=-1):

        if sigma < 0:
            sigma_est   = restoration.estimate_sigma(nuclear_frame)
        else:
            sigma_est   = sigma

        filtered= ndi.gaussian_filter(nuclear_frame,0.5*sigma_est)

        seed    = np.copy(filtered,1)
        seed[1:-1,1:-1] = filtered.min()
        mask    = np.copy(filtered)

        dilated = morphology.reconstruction(seed,mask,method='dilation')
        bgsub   = filtered - dilated

        self.lit_pxl = np.where(bgsub > np.average(bgsub))
        self.lit_crd = np.vstack((self.lit_pxl[0],self.lit_pxl[1])).T

        self.clusters= DBSCAN(eps=np.sqrt(2),min_samples=2).fit(self.lit_crd)
        nuc_path, nuc_points, nuc_fill = self.findNucEdgeFromClusters(nuclear_frame)
        
        if self.ts==0:
            self.saveNuclearMask(nuc_points)

        return nuc_path, nuc_points, nuc_fill


    def saveNuclearMask(self,nuc_points):
        ofile = open(self.output_prefix+"NuclMask.txt",'w')
        for idx in range(len(nuc_points)):
            ofile.write(str(nuc_points[idx][0])+","\
                       +str(nuc_points[idx][1])+"\n")
        ofile.close()

    def findNucEdgeFromClusters(self,nuclear_frame):
        #Assume that the cell is in the center of the frame
        Nx = len(nuclear_frame)
        Ny = len(nuclear_frame[0])
        
        xMid = int(Nx/2)
        yMid = int(Ny/2)
        

        #Find the edges of each cluster
        for idx in range(np.max(self.clusters.labels_)+1):
            blank = np.zeros_like(nuclear_frame)
            members = np.where(self.clusters.labels_ == idx)[0]
            x = self.lit_crd[members,0]
            y = self.lit_crd[members,1]
            
            xbounds = np.array([np.min(x),np.max(x)])
            ybounds = np.array([np.min(y),np.max(y)])
            for x_idx in range(xbounds[0],xbounds[1]+1):
                these_x = np.where(x == x_idx)[0]
                if len(these_x) > 0:
                    min_y   = np.min(y[these_x])
                    max_y   = np.max(y[these_x])
                    try:
                        lower_bound= np.vstack((lower_bound,[x_idx,min_y]))
                    except:
                        lower_bound= np.array([x_idx,min_y])
                    try:
                        upper_bound= np.vstack(([x_idx,max_y],upper_bound))
                    except:
                        upper_bound= np.array([x_idx,max_y])
                else:
                    print("Warning: No X values in this lane: "+str(x_idx))
            nuc_points = np.vstack((lower_bound,upper_bound))
            nuc_points = np.vstack((nuc_points,nuc_points[0]))

            for idx2 in range(len(x)):
                blank[x[idx2],y[idx2]] = 1
                
            nuc_path = Path(nuc_points,closed=True)

            if nuc_path.contains_point([xMid,yMid]):
                break
            else:
                del nuc_points
                del upper_bound
                del lower_bound

        return nuc_path,nuc_points,blank


    def growROI(self,roi_file,roi_buffer):
        #Store Path objects for all the requested ROIs for later callback
        roi_path_dict = {}
        roi_frame_dict= {}

        roi_pix  = np.where(roi_file>0)
        row_start= np.min(roi_pix[0]) - roi_buffer[1]
        row_end  = np.max(roi_pix[0]) + roi_buffer[1]
        col_start= np.min(roi_pix[1]) - roi_buffer[0]
        col_end  = np.max(roi_pix[1]) + roi_buffer[0]
       
        roi_height = row_end - row_start 
        roi_width  = col_end - col_start - 1

        for idx in range(-self.additional_rois,self.additional_rois+1):

            rowStart= row_start+idx*(roi_height) 
            rowEnd  = row_end+idx*(roi_height)

            colStart= col_start
            colEnd  = col_end

            out_roi_file= np.zeros(np.shape(roi_file))
            out_roi_file[rowStart:rowEnd+1,colStart:colEnd+1] = 1

            roi_frame_dict[idx] = np.copy(out_roi_file)

            roi_path = Path(np.array([[colStart,rowStart],
                                      [colStart,rowEnd],
                                      [colEnd,rowEnd],
                                      [colEnd,rowStart],
                                      [colStart,rowStart]],dtype=int))
            roi_path_dict[idx] = roi_path
             
            if idx==0:
                ofile = open(self.output_prefix+"ROI.txt",'w')
                ofile.write(str(colStart+1)+","+str(rowStart+1)+",")
                ofile.write(str(roi_width)+",")
                ofile.write(str(roi_height)+"\n")
                ofile.close()

        return roi_frame_dict,roi_path_dict

    def correctBackground(self,input_frame,input_video,channel,is_nd2=False):
        if self.ts == 0:
            for idx in range(self.irrad_frame):
                if is_nd2:
                    this_frame = input_video.get_frame_2D(c=channel,t=idx)
                else:
                    this_frame = np.array(bioformats.load_image(input_video,c=self.protein_channel,t=self.ts,rescale=False))
                self.histogram,bins = np.histogram(this_frame,
                              bins=np.arange(1,np.max(this_frame)+1))
        
                #Smooth the histogram out
                temp    = np.convolve(self.histogram,
                                  np.ones(3,dtype=int),'valid')
                ranges  = np.arange(1,2,2)
                start   = np.cumsum(self.histogram[:2])[::2]/ranges
                stop    = (np.cumsum(self.histogram[:-3:-1])[::2]/ranges)[::-1]
                self.smoothed_hist  = np.concatenate((start,temp,stop))
                self.peaks    = scipy.signal.find_peaks(self.smoothed_hist)[0]
                self.min_peak_int = np.max(self.smoothed_hist[self.peaks])
                self.min_peak = np.where(self.smoothed_hist==self.min_peak_int)[0][0]
                self.bg_value       = bins[np.where(
                                      self.smoothed_hist[self.min_peak:]<=\
                                      (0.67*self.min_peak_int))[0][0]]\
                                      +self.min_peak
                self.avg_bg         = np.average(this_frame[np.where(
                                      input_frame<=self.bg_value)])

                try:
                    self.arr_of_avgs = np.append(self.arr_of_avgs,
                                                 self.avg_bg)
                except:
                    self.arr_of_avgs = np.array([self.avg_bg])

            self.bg_correction  = np.average(self.arr_of_avgs)

        output_frame = input_frame - self.bg_correction
        output_frame[np.where(output_frame < 0)] = 0
        return output_frame
#############################################################################################


##############################################################################################
class OtherReader(bioformats.ImageReader):
    def md(self):
        if self._md is None:
            self._md = []
            _io = np.arange(self.series_count)
            idxs= np.arange(self.series_count)
            for _i in idxs:
                self.rdr.setSeries(_i)
                series_md = javabridge.jutil.jdictionary_to_string_dictionary(self.rdr.getSeriesMetadata())
                self._md.append(series_md)
            self.rdr.setSeries(_io)
        return self._md
