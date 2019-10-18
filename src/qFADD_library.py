#!/usr/bin/env python
import datetime
import numpy as np
import math
import itertools
import argparse
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore") #Suppress matplotlib tight_layout() warning
from matplotlib.path import Path
from scipy.interpolate import interp1d as interpolate
from scipy.stats import linregress
import time
import os
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
##### MPI ENVIRONMENT #####
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
###########################

class NuclearDiffusion(object):
    def __init__(self,parent = None):
        if rank==0:
            print("Q-FADD model initialized on "+str(size)+" processors.")
        pass

    def main(self,input_object):
        ''' Runs the Code '''
        if rank==0:
            self.qfadd_start = time.time()
        self.parse_inputs(input_object)
        self.diffusion_model()
        if rank==0:
            rmsdfile= self.output_prefix+"_rmsd_matrix.dat"
            r2file  = self.output_prefix+"_r2_matrix.dat"
            np.savetxt(rmsdfile,self.rmsd_matrix,fmt='%.4e',delimiter=' ')
            np.savetxt(r2file,self.r2_matrix,fmt='%.4e',delimiter=' ')
            self.qfadd_end = time.time()
            elap_time = self.qfadd_end - self.qfadd_start
            elap_time = np.around(elap_time/60.,decimals=2)
            print("\n\nProgram complete in "+str(elap_time)+" minutes.")

    def diffusion_model(self):
        #Make all possible combination of D and mobile values
        if rank==0:
            self.D_mobile_sets  = list(itertools.product(self.d_range,self.m_range))
            self.n_sets         = len(self.D_mobile_sets)
            print("Comparing the following "+str(self.n_sets)+" combinations "\
                 +"([D_value,mobile_value]): "+str(self.D_mobile_sets))
            for thread in range(1,size):
                comm.send(self.D_mobile_sets,dest=thread,tag=0)
        else:
            self.D_mobile_sets  = comm.recv(source=0,tag=0)
            self.n_sets = len(self.D_mobile_sets)

        self.best_r_squared = -99999.999 #Initialize r-square to very low value (good r^2 ~= 1.0)
        self.best_rmsd      = 999999.999 #Initialize rmsd to very high value (good rmsd ~= 0.0)

        for combo_idx in range(rank,self.n_sets,size):
            self.this_D     = self.D_mobile_sets[combo_idx][0]
            self.this_mobile= self.D_mobile_sets[combo_idx][1]
            self.r2_track   = np.zeros(self.ensemble_size,dtype=float)
            self.rmsd_track = np.zeros(self.ensemble_size,dtype=float)
            self.ensemble_timeseries= np.zeros((self.ensemble_size,self.nsteps),dtype=float)
            self.ensemble_init_roi0 = np.zeros(self.ensemble_size,dtype=float)
            for self.ensemble_idx in range(self.ensemble_size):
                self.idx = 0 #Initialize each ensemble step to 0
                self.image_array= []
                self.build_model_nucleus()
                init_time = time.time()
                self.count_roi0_intensity()
                self.is_mobile  = np.zeros(self.particles,dtype=int)
                self.is_mobile  = np.zeros(self.particles,dtype=int)
                for particle in range(self.particles):
                    if np.random.randint(1000) < self.this_mobile:
                        self.is_mobile[particle] = 1
                mobiles = np.where(self.is_mobile==1)[0]
                for self.idx in range(1,self.nsteps):
                    for particle in mobiles:
                        self.propagate(particle)
                    if self.save_movie:
                        self.plot_points_within_nucleus()
                    self.count_roi0_intensity()
                self.interpolate_diff_model()
                self.calc_r_squared(printout=False)
                self.r2_track[self.ensemble_idx]    = self.r_squared
                 
                self.calc_rmsd(printout=False)
                self.rmsd_track[self.ensemble_idx]  = self.rmsd
                
                #if self.plot_all:
                #    self.plot_roi_intensity_timeseries(gridpoint=True)
                
                self.ensemble_timeseries[self.ensemble_idx] = np.copy(self.num_in_roi0)
                self.ensemble_init_roi0[self.ensemble_idx]  = np.copy(self.initial_num_in_roi0)
                end_time = time.time()
                print("rank "+str(rank)+": Simulation for (D="\
                      +str(self.this_D)+" px/step,mobile="\
                      +str(self.this_mobile)+" ppt, replica="\
                      +str(self.ensemble_idx)+", rmsd="\
                      +str(np.around(self.rmsd,decimals=4))+", r^2="\
                      +str(np.around(self.r_squared,decimals=4))+") took "\
                      +str(end_time - init_time)+" seconds.")

                if self.save_movie:
                    self.make_movie()

            if self.use_avg:
                norm_ensemble_timeseries = np.copy(self.ensemble_timeseries)
                for idx in range(self.ensemble_size):
                    norm_ensemble_timeseries[idx] = self.ensemble_timeseries[idx]/self.ensemble_init_roi0[idx]
                
                self.average_timeseries = np.average(norm_ensemble_timeseries,axis=0)
                model_int,model_res     = self.interpolate_diff_model(model_int=self.average_timeseries)
                
                med_r2   = self.calc_r_squared(return_val=True,model_int=model_int)
                med_rmsd = self.calc_rmsd(return_val=True,model_int=model_int,model_res=model_res)
                med_model= self.average_timeseries
                norm_model  = np.copy(med_model)
                med_roi0 = 1.0
            else:
                med_r2   = np.median(self.r2_track)
                med_rmsd = np.median(self.rmsd_track)
                if self.ensemble_size > 1:
                    med_idx  = np.where(self.rmsd_track==med_rmsd)[0][0]
                else:
                    med_idx  = 0
                med_model= self.ensemble_timeseries[med_idx]
                med_roi0 = self.ensemble_init_roi0[med_idx]
                norm_model  = med_model/med_roi0
                model_int,model_res     = self.interpolate_diff_model(model_int=norm_model)

            if self.plot_all:
                self.plot_roi_intensity_timeseries(gridpoint=True,model_int=norm_model,model_res=model_res)


            if combo_idx == rank:
                best_r2     = np.copy(med_r2)
                best_rmsd   = np.copy(med_rmsd)
                self.best_timeseries= np.copy(norm_model)
                #self.best_init_roi0 = np.copy(med_roi0)
                self.best_residuals = np.copy(model_res)
            else:
                if med_rmsd < best_rmsd:
                    best_r2     = np.copy(med_r2)
                    best_rmsd   = np.copy(med_rmsd)
                    self.best_timeseries= np.copy(norm_model)
                    #self.best_init_roi0 = np.copy(med_model)
                    self.best_residuals = np.copy(model_res)
            d_idx    = np.where(self.d_range==self.this_D)[0][0]
            m_idx    = np.where(self.m_range==self.this_mobile)[0][0]

            self.r2_matrix[d_idx,m_idx]  = med_r2
            self.rmsd_matrix[d_idx,m_idx]= med_rmsd

        if rank==0:
            #Upper range limit corrects for if too many cpus are asked for
            for thread in range(1,np.min([size,len(self.D_mobile_sets)])):
                self.r2_matrix  += comm.recv(source=thread,tag=1)
                self.rmsd_matrix+= comm.recv(source=thread,tag=2)
                dum_rmsd        = comm.recv(source=thread,tag=3)
                dum_timeseries  = comm.recv(source=thread,tag=4)
                #dum_norm_val    = comm.recv(source=thread,tag=5)
                dum_residuals   = comm.recv(source=thread,tag=5)
                if dum_rmsd < best_rmsd:
                    best_rmsd = dum_rmsd
                    self.best_timeseries= np.copy(dum_timeseries)
                    #self.best_init_roi0 = np.copy(dum_norm_val)
                    self.best_residuals = np.copy(dum_residuals)
            self.normalized_roi0_intensity = self.best_timeseries
            #self.num_in_roi0        = self.best_timeseries
            #self.initial_num_in_roi0= self.best_init_roi0
            #self.normalized_roi0_intensity = self.num_in_roi0 / self.initial_num_in_roi0
            self.interpolate_diff_model()
            self.calc_r_squared(printout=False)
            self.calc_rmsd(printout=False)
            self.print_intensity_timeseries()
            self.plot_r2_map()
            self.plot_rmsd_map()
            self.report_best_model()
        elif rank < len(self.D_mobile_sets): #Correct for if too many cpus are asked for
            comm.send(self.r2_matrix,dest=0,tag=1)
            comm.send(self.rmsd_matrix,dest=0,tag=2)
            comm.send(best_rmsd,dest=0,tag=3)
            comm.send(self.best_timeseries,dest=0,tag=4)
            #comm.send(self.best_init_roi0,dest=0,tag=5)
            comm.send(self.best_residuals,dest=0,tag=5)
        else:
            print("Rank "+str(rank)+": Too many CPUs requested.  This thread does not contribute to analysis.")

    def parse_inputs(self,input_object):
        '''Store values from argparse internally'''
        self.mask_file      = input_object.mask_file
        self.roi_file       = input_object.roi_file
        self.kinetics_file  = input_object.kinetics_file

        #Make a directory for the nucleus processing
        self.output_dir     = os.path.splitext(os.path.basename(self.mask_file))[0]
        if not os.path.isdir(self.output_dir):
            try:
                os.mkdir(self.output_dir)
            except:
                pass

        self.output_prefix  = os.path.join(self.output_dir,
                                           input_object.output_prefix)
        
        self.t0             = input_object.t0           #Offset for times such that the experiment and model line up
        self.norm_frames    = input_object.norm_frames  #Number of frames to use for normalization
        #self.column_start   = input_object.column_start #Starting column for drawing ROI data from
        #self.column_end     = input_object.column_end   #Endind column for ROI data
        self.roi0_column    = input_object.roi0_column
        self.mobile_min     = input_object.mobile_min 	#Minimum number of mobile particles for grid search
        self.mobile_max     = input_object.mobile_max 	#Maximum number of mobile particles for grid search
        self.mobile_step    = input_object.mobile_step	#Step size in mobile particles for grid search
        self.scale          = input_object.scale        #Pixel size (in micro-meters)
        self.tstep          = input_object.tstep        #timestep (in seconds)
        self.nsteps         = input_object.nsteps       #Number of iterations; sim_time = tstep * nsteps
        self.particles      = input_object.particles    #Total number of particles
        self.Dmin	        = input_object.Dmin 	    #Minimum value of diffusion coefficient for grid search
        self.Dmax	        = input_object.Dmax		    #Maximum value of diffusion coefficient for grid search
        self.Dstep	        = input_object.Dstep	    #Step size in D for grid search
        self.ensemble_size  = input_object.ensemble_size#Run multiple copies of diffusion model to determine accurate ensemble model
        self.roi_radius     = input_object.roi_radius   #Radius for Theoretical Circular DNA damage zone
        self.find_plat      = input_object.find_plat    #Determine simulation length from when (if) experimental intensity plateaus
        self.use_f_formula  = input_object.use_f_formula#Use Johannes' Formula for approximating mobile fraction instead of grid searching
        self.save_movie     = input_object.save_movie   #Boolean flag to make movie of simulation (significantly slows program)
        self.use_avg        = input_object.use_avg      #Calculate avg sampled trajectory rather than pick median sampled per grid point
        self.plot_all       = input_object.plot_all     #Plot the intensity timeseries for every gridpoint, not just the best model


        if self.plot_all:
            self.plot_dir   = os.path.join(self.output_dir,input_object.output_prefix+"_gridpoint_fits")
            if rank==0:
                if not os.path.isdir(self.plot_dir):
                    try:
                        os.mkdir(self.plot_dir)
                    except:
                        pass

        self.scaledDmin     = 0.5 * self.Dmin**2 * self.scale**2 * (1./self.tstep)
        self.scaledDmax	    = 0.5 * self.Dmax**2 * self.scale**2 * (1./self.tstep)
        self.scaledDstep    = 0.5 * self.Dstep**2 * self.scale**2 * (1./self.tstep)        

        self.idx            = 0     #Initialize first iteration (frame) to 0

        self.convert_inputs()
        self.save_inputs()
        	
        if rank==0:
            print("===========================================")
            print("Will conduct grid search protocol using the following limits:")
            print("\t-Minimum Diffusion Coefficient: "+str(np.around(self.scaledDmin,decimals=4))+" um^2 per second")
            print("\t-Maximum Diffusion Coefficient: "+str(np.around(self.scaledDmax,decimals=4))+" um^2 per second")
            print("\t-Diffusion Grid Spacing: "+str(self.scaledDstep)+" um^2 per second\n")
            print("\t-Minimum Mobile Fraction: "+str(self.mobile_min)+" molecules per thousand")
            print("\t-Maximum Mobile Fraction: "+str(self.mobile_max)+" molecules per thousand")
            print("\t-Mobile Fraction Grid Spacing: "+str(self.mobile_step)+" molecules per thousand\n") 
            print("Current parameters will create a model trajectory of "+str(np.around(self.tstep * self.nsteps,decimals=4))+" seconds")
            print("===========================================")
        
    def save_inputs(self):
        out_filename = self.output_prefix+"_inputs.par"
        ofile = open(out_filename,'w')
        ofile.write("#Input parameters for run on "\
                   +str(datetime.datetime.now())\
                   +"\n")
        ofile.write("#Analysis used "+str(size)+" processors.\n\n")

        ofile.write("Mask File              : "\
                   +self.mask_file+"\n")
        ofile.write("ROI File               : "\
                   +self.roi_file+"\n")
        ofile.write("Kinetics File          : "\
                   +self.kinetics_file+"\n")
        ofile.write("Output Prefix          : "\
                   +os.path.basename(self.output_prefix)\
                   +"\n")
        ofile.write("Offset time            : "\
                   +str(self.t0)+" s\n")
        ofile.write("Normalization Frames   : "\
                   +str(self.norm_frames)+" frames\n")
        ofile.write("Number of Molecules    : "\
                   +str(self.particles)+" molecules\n")
        ofile.write("Pixel Resolution       : "\
                   +str(self.scale)+" um/pixel\n")
        ofile.write("Simulation Timestep    : "\
                   +str(self.tstep)+" s\n")
        ofile.write("Minimum Mobile Fraction: "\
                   +str(self.mobile_min)+" ppt\n")
        ofile.write("Maximum Mobile Fraction: "\
                   +str(self.mobile_max)+" ppt\n")
        ofile.write("Mobile Fraction Stride : "\
                   +str(self.mobile_step)+" ppt\n")
        ofile.write("Minimum Diff. Constant : "\
                   +str(self.Dmin)+" pixels/step\n")
        ofile.write("Maximum Diff. Constant : "\
                   +str(self.Dmax)+" pixels/step\n")
        ofile.write("Diff. Constant Stride  : "\
                   +str(self.Dstep)+" pixels/step\n")
        ofile.write("Ensemble Size          : "\
                   +str(self.ensemble_size)+" replicas\n")
        ofile.write("Sim. Length from Exp.? : "\
                   +str(self.find_plat)+"\n")
        ofile.write("Plot every point?      : "\
                   +str(self.plot_all)+"\n")
        ofile.close()



    def convert_inputs(self):
        '''Convert argparse to real data'''
        self.rdata      = np.genfromtxt(self.mask_file,dtype=int,delimiter=',')
        self.roi_data   = np.genfromtxt(self.roi_file,dtype=int,delimiter=',')
        self.kdata      = np.genfromtxt(self.kinetics_file,skip_header=4,dtype=float,delimiter=',')
        
        self.center_mask()
        if self.roi_radius==0:
            self.roi_data_to_coords()
        else:
            self.theoretical_circle_roi()
        
        self.exp_times  = self.kdata[:,0]
        #self.exp_times  = self.exp_times - self.t0
        self.exp_times  = np.reshape(self.exp_times,(len(self.exp_times),1))    #Re-shape so that can stack with intensity data

        #self.roi0_column= int((self.column_start - self.column_end)/2.0)
        #self.exp_data   = self.kdata[:,self.column_start:self.column_end]                       #Select only the channel you are interested in
        self.exp_data   = self.kdata[:,self.roi0_column] #Select only the channel you are interested in
        self.exp_data   = np.divide(self.exp_data,np.average(self.exp_data[:self.norm_frames]))   #Normalize the intensities
        self.exp_data   = np.hstack((self.exp_times,np.reshape(self.exp_data,(len(self.exp_data),1))))

        #Extrapolate the experimental intensities to intersect I = 1 to determine offset time
        if self.t0 == -1:
            self.extrapolate_offset()
        else:
            self.extrapolated_offset = self.t0
        self.exp_data[:,0] = self.exp_data[:,0] - self.extrapolated_offset
        self.exp_times  = self.exp_times - self.extrapolated_offset

        #Determine number of steps from intensity timeseries maximum point
        if self.find_plat:
            self.find_plateau()

        #Determine if model time is longer or shorter than experiment
        self.model_time = np.arange(self.nsteps) * self.tstep
        max_exp_time    = self.exp_times[-1]
        max_model_time  = self.model_time[-1]
        #If experimental time is longer
        if max_exp_time > max_model_time:
            time_high_cut   = np.where(self.exp_times > max_model_time)[0][0]
            time_low_cut    = np.where(self.exp_times >= 0)[0][0]
            self.cut_exp_time   = self.exp_times[time_low_cut:time_high_cut]
            self.cut_exp_roi0   = self.exp_data[time_low_cut:time_high_cut,1]#,self.roi0_column]
            self.plot_exp_time  = self.exp_times[:time_high_cut]
            self.plot_exp_roi0  = self.exp_data[:time_high_cut,1]#,self.roi0_column]
            self.cut_model_time = np.copy(self.model_time)
            self.cut_model_idx  = np.arange(self.nsteps)
        #If the requested model time is longer than experiment
        else:
            time_high_cut       = np.where(self.model_time > max_exp_time)[0][0]
            time_low_cut        = np.where(self.exp_times >= 0)[0][0]
            self.cut_exp_time   = self.exp_times[time_low_cut:]
            self.cut_exp_roi0   = self.exp_data[time_low_cut:,1]#,self.roi0_column]
            self.plot_exp_time  = np.copy(self.exp_times)
            self.plot_exp_roi0  = np.copy(self.exp_data[:,1])#,self.roi0_column])
            self.cut_model_time = np.copy(self.model_time[:time_high_cut])
            self.cut_model_idx  = np.arange(self.nsteps)[:time_high_cut]

        if self.use_f_formula:
            try:
                self.approximate_mobile_fraction()
            except:
                print("WARNING: Mobile fraction approximator not currently functional!  Defaulting to full grid search!")
                self.m_range= np.arange(self.mobile_min,self.mobile_max+1,self.mobile_step)
        else:
            self.m_range= np.arange(self.mobile_min,self.mobile_max+1,self.mobile_step)

        self.d_range    = np.arange(self.Dmin,self.Dmax+1,self.Dstep)
        self.d_range_um = 0.5 * np.power(self.d_range,2) * self.scale**2 * (1./self.tstep)

        self.n_d        = len(self.d_range)
        self.n_m        = len(self.m_range)

        self.r2_matrix  = np.zeros((self.n_d,self.n_m),dtype=float)
        self.rmsd_matrix= np.zeros((self.n_d,self.n_m),dtype=float)

    def extrapolate_offset(self):
        #Don't include normalization frames to spline fitting
        exp_data_for_spline = self.exp_data[self.norm_frames:]#,
                                            #[0,self.roi0_column]]
        spline_function     = interpolate(exp_data_for_spline[:,0],
                                          exp_data_for_spline[:,1],
                                          fill_value='extrapolate')
        extrapolated_times  = np.arange(np.min(self.exp_times),
                                        np.max(self.exp_times),
                                        0.05)
        extrapolated_curve  = np.around(spline_function(extrapolated_times),
                                        decimals = 2)
        extrapolated_idx    = int(np.median(
                                  np.where(extrapolated_curve==1.00)[0]))
        intersect_time      = np.around(extrapolated_times[extrapolated_idx],
                                        decimals = 3)
        self.extrapolated_offset = np.copy(intersect_time)

        if 0:
            plt.figure(figsize=(6.,4.))
            plt.plot(self.exp_data[:,0],self.exp_data[:,1],label='Exp.',marker='.',linestyle='')#self.roi0_column],label='Exp.',marker='.',linestyle='')
            plt.plot(exp_data_for_spline[:,0],exp_data_for_spline[:,1],marker='.',linestyle='',color='red')
            plt.plot(extrapolated_times,extrapolated_curve,label='Extrapolated')
            plt.axvline(extrapolated_times[extrapolated_idx],linestyle='--',color='k')
            plt.legend(loc=0)
            plt.savefig('intensity_extrapolation.pdf',format='pdf')

    def center_mask(self):
        '''Moves nuclear mask to center at (0,0)'''
        x_center    = np.around(np.mean(self.rdata[:,0]))
        y_center    = np.around(np.mean(self.rdata[:,1]))

        self.com    = np.array([x_center,y_center],dtype=int)

        self.rdata[:,0] = np.subtract(self.rdata[:,0],self.com[0])
        self.rdata[:,1] = np.subtract(self.rdata[:,1],self.com[1])
        self.rdata      = np.vstack((self.rdata,self.rdata[0]))

        self.nuc_mask_path_obj  = Path(self.rdata[:,::-1],closed=True)

    def roi_data_to_coords(self):
        '''Makes ROI on top of nuclear mask'''
        #self.n_roi  = int((self.column_end - self.column_start)/2)
        self.n_roi  = 0
        roi_data    = self.roi_data
        roi_box     = np.zeros((5,2),dtype=int)
        roi_box[0,0]= roi_data[0]
        roi_box[0,1]= roi_data[1]
        roi_box[1,0]= roi_data[0] + roi_data[2]
        roi_box[1,1]= roi_data[1]
        roi_box[2,0]= roi_box[1,0]
        roi_box[2,1]= roi_data[1] + roi_data[3]
        roi_box[3,0]= roi_data[0]
        roi_box[3,1]= roi_box[2,1]
        roi_box[4,0]= roi_box[0,0]
        roi_box[4,1]= roi_box[0,1]

        self.roi_coords = np.copy(roi_box)
        self.roi_coords = self.roi_coords - self.com[::-1]

        self.roi_dict   = dict()
        self.roi_dict[0]= self.roi_coords

        
        for flank in range(self.n_roi+1):
            self.roi_dict[flank] = np.copy(self.roi_coords)
            self.roi_dict[flank][:,1] += (flank * self.roi_data[3])
            
            self.roi_dict[-flank]= np.copy(self.roi_coords)
            self.roi_dict[-flank][:,1]+= (-flank * self.roi_data[3])

        self.roi0_path_obj  = Path(self.roi_dict[0],closed=True)

    def theoretical_circle_roi(self):
        '''Makes a circular ROI to test shape of ROI dependence'''
        r = self.roi_radius
        range_of_angles = np.arange(0,2*np.pi,2*np.pi/100)
        N = len(range_of_angles)
        self.roi_coords = np.zeros((N,2),dtype=float)
        self.roi_coords[:,0] = r * np.cos(range_of_angles)
        self.roi_coords[:,1] = r * np.sin(range_of_angles)

        self.roi0_path_obj  = Path(self.roi_coords,closed=True)


    def build_model_nucleus(self):
        self.roi_figure = plt.figure(figsize=(5.0,5.0))
        ax = self.roi_figure.add_subplot(111)
        plt.title('ROI Overlay')
        plt.plot(self.rdata[:,1],self.rdata[:,0],color='grey')

        if self.roi_radius == 0:
            for idx in range(-(self.n_roi),self.n_roi+1):
                if idx < 0:
                    color   = 'blue'
                elif idx==0:
                    color   = (0.1,1.0,0.1)
                else:
                    color   = 'red'
                x_values    = np.unique(self.roi_dict[idx][:,0])
                y_values    = np.unique(self.roi_dict[idx][:,1])
                plt.fill_between(x_values,y_values[0],y_values[1],facecolor=color,alpha=0.5*(1-np.abs(idx)/5.),linewidth=0.0)
        else:
            roi0_patch = patches.PathPatch(self.roi0_path_obj,color=(0.1,1.0,0.1))
            ax.add_patch(roi0_patch)
            plt.scatter(self.roi_coords[:,0],self.roi_coords[:,1],c=np.arange(len(self.roi_coords[:,1])))
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.xlim(-200,200)
        plt.ylim(-200,200)
        if rank==0:
            plt.tight_layout()
            plt.savefig(self.output_prefix+"_Model_Nucleus_with_ROI.pdf",format='pdf')
        plt.close('all')
        self.make_points_within_nucleus()
         
    def make_points_within_nucleus(self):
        xmin    = np.min(self.rdata[:,0])
        ymin    = np.min(self.rdata[:,1])
        xmax    = np.max(self.rdata[:,0])
        ymax    = np.max(self.rdata[:,1])

        self.points_x    = np.around(np.random.uniform(low=xmin,high=xmax+1e-8,size=self.particles))
        self.points_y    = np.around(np.random.uniform(low=ymin,high=ymax+1e-8,size=self.particles))
        
        self.point_tracker  = np.zeros(self.particles,dtype=int)

        for point in range(self.particles):
            self.point_tracker[point]   = self.check_inside(point)
            while self.point_tracker[point]==0:
                new_x   = np.around(np.random.uniform(low=xmin,high=xmax+1e-8))
                new_y   = np.around(np.random.uniform(low=ymin,high=ymax+1e-8))
                self.points_x[point]    = new_x
                self.points_y[point]    = new_y
                self.point_tracker[point]   = self.check_inside(point)

        if self.save_movie:
            self.plot_points_within_nucleus()

    def plot_points_within_nucleus(self):
        fig = plt.figure(figsize=(5.0,5.0))
        plt.plot(self.rdata[:,1],self.rdata[:,0],color='grey')
        
        in_points   = np.where(self.point_tracker==1)[0]
        out_points  = np.where(self.point_tracker==0)[0]
        roi_points  = np.where(self.point_tracker==2)[0]

        plt.plot(self.points_y[in_points],self.points_x[in_points],color='blue',marker='.',markersize=4,linestyle='',label='N = '+str(len(in_points)),alpha=0.20)
        plt.plot(self.points_y[roi_points],self.points_x[roi_points],color=(.1,1.,.1),marker='.',markersize=4,linestyle='',label='N = '+str(len(roi_points)),alpha=0.20)
        if len(out_points) > 0:
            plt.plot(self.points_y[out_points],self.points_x[out_points],color='red',marker='.',markersize=4,linestyle='',label='N = '+str(len(out_points)),alpha=0.2)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.xlim(-200,200)
        plt.ylim(-200,200)
        plt.tight_layout()
        image_string = self.output_prefix + "_D_"+str(self.this_D)\
                     + "_mobile_"+str(self.this_mobile)+"_simulation_"\
                     + str(self.ensemble_idx) + "_frame%04d" % (self.idx,)\
                     + ".png"
        plt.savefig(image_string,format='png',dpi=400)
        plt.close('all')
        self.image_array.append(image_string)

    def check_inside(self,point):
        this_x  = self.points_x[point]
        this_y  = self.points_y[point]
        
        if self.roi0_path_obj.contains_point([this_y,this_x]):
            return 2
        elif self.nuc_mask_path_obj.contains_point([this_y,this_x]):
            return 1
        else:
            return 0

    def draw_D1_value(self):
        D1 = np.around(np.random.normal(loc=self.meanD1,scale=self.stddevD1))
        return D1

    def propagate(self,this_point):
        #D_this_step = self.draw_D1_value()
        D_this_step = self.this_D
        #if (self.point_tracker[this_point] != 2):
        if self.roi_radius==0:
            if not ((self.points_x[this_point] > self.roi_coords[0,1]) and (self.points_x[this_point] < self.roi_coords[2,1])):
                old_x = self.points_x[this_point]
                old_y = self.points_y[this_point]
            

                self.points_x[this_point] += np.around(D_this_step * np.random.choice([-1,1]))
                self.points_y[this_point] += np.around(D_this_step * np.random.choice([-1,1]))

                is_inside = self.check_inside(this_point)
                if is_inside == 0:
                    self.points_x[this_point] = old_x
                    self.points_y[this_point] = old_y
                else:
                    self.point_tracker[this_point] = is_inside
        else:
            if not self.roi0_path_obj.contains_point([self.points_y[this_point],self.points_x[this_point]]):
                old_x = self.points_x[this_point]
                old_y = self.points_y[this_point]

                self.points_x[this_point] += np.around(D_this_step * np.random.choice([-1,1]))
                self.points_y[this_point] += np.around(D_this_step * np.random.choice([-1,1]))

                is_inside = self.check_inside(this_point)
                if is_inside == 0:
                    self.points_x[this_point] = old_x
                    self.points_y[this_point] = old_y
                else:
                    self.point_tracker[this_point] = is_inside


    def count_roi0_intensity(self):
        if self.idx == 0:
            self.initial_num_in_roi0= len(np.where(self.point_tracker==2)[0])
            self.num_in_roi0        = np.zeros(self.nsteps,dtype=int)
            self.normalized_roi0_intensity  = np.zeros(self.nsteps,dtype=float)
        self.num_in_roi0[self.idx]  = len(np.where(self.point_tracker==2)[0])
        self.normalized_roi0_intensity[self.idx]    = self.num_in_roi0[self.idx] / self.initial_num_in_roi0

    def interpolate_diff_model(self,model_int=None):
        if model_int is None:
            model_timeseries = self.normalized_roi0_intensity
        else:
            model_timeseries = model_int

        interpolated_function = interpolate(self.cut_model_time,model_timeseries[self.cut_model_idx])

        interpolated_model  = interpolated_function(self.cut_exp_time)
        interpolated_model  = np.reshape(interpolated_model,np.shape(self.cut_exp_roi0))

        residuals   = np.subtract(interpolated_model,self.cut_exp_roi0)

        if model_int is None:
            self.interpolated_model = np.copy(interpolated_model)
            self.residuals          = np.copy(residuals)
        else:
            return interpolated_model,residuals


    def calc_r_squared(self,printout=True,return_val=False,model_int=None):
        if model_int is None:
            model_int = self.interpolated_model
        else:
            pass
            
        SSres   = 0.
        SStot   = 0.
        mean    = 0.
        i       = len(self.cut_exp_roi0)

        for idx in range(len(model_int)):
            SSres += (self.cut_exp_roi0[idx] - model_int[idx])**2
            mean  += self.cut_exp_roi0[idx]

        mean = mean/i

        for idx in range(len(self.interpolated_model)):
            SStot += (self.cut_exp_roi0[idx] - mean)**2

        self.r_squared = 1 - (SSres/SStot)

        print("\n\n")
        if printout: 
            print("R-squared of the fit: "+str(np.around(self.r_squared,decimals=5)))

        if return_val:
            return self.r_squared

    def calc_rmsd(self,printout=True,return_val=False,model_int=None,model_res=None):
        if model_int is None:
            model_int   = np.copy(self.interpolated_model)
            model_res   = np.copy(self.residuals)
        else:
            if model_res is None:
                model_res   = np.subtract(model_int,self.cut_exp_roi0)
            else:
                pass

        sq_deviation        = np.power(model_res,2)
        mean_sq_deviation   = np.average(sq_deviation)
        self.rmsd           = np.sqrt(mean_sq_deviation)
    
        if printout:    
            print("RMSD of the fit: "+str(np.around(self.rmsd,decimals=5)))

        if return_val:
            return self.rmsd
        #Weight each point by normalizing with the measured intensity value (weighted by 1/[exp intensity])
        #deviation_wt        = np.divide(self.deviation,self.cut_exp_roi0)
        #sq_deviation_wt     = np.power(deviation_wt,2)
        #mean_sq_deviation_wt= np.average(sq_deviation_wt)
        #self.rmsd_weighted  = np.sqrt(mean_sq_deviation_wt)

        #print("Weighted RMSD of the fit: "+str(np.around(self.rmsd_weighted,decimals=5)))

    def plot_roi_intensity_timeseries(self,gridpoint=False,model_int=None,model_res=None):

        if model_int is None:
            model_int = np.copy(self.normalized_roi0_intensity)
            model_res = np.copy(self.residuals)
        else:
            if model_res is None:
                model_res   = np.subtract(model_int,self.cut_exp_roi0)

        if rank==0 or gridpoint==True:
            plt.figure(figsize=(6.0,4.0))
            gs  = matplotlib.gridspec.GridSpec(2,1,height_ratios=[3,1])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
        
            #only plot model times that have corresponding experiment times
            #max_time        = self.exp_times[-1]
            #max_model_time  = self.model_time[-1]
            #if max_time > max_model_time:
            #    exp_cut = np.where(self.exp_times > max_model_time)[0][0]
            #    exp_time_plot = self.exp_times[:exp_cut]
            #    exp_roi0_plot = self.exp_data[:exp_cut,self.roi0_column]
    
                #model_roi0_plot = self.normalized_roi0_intensity
            #    model_roi0_plot = np.copy(model_int)
            #else:
            #    exp_time_plot = self.exp_times
            #    exp_roi0_plot = self.exp_data[:,self.roi0_column]

            #    model_over_idx = np.where(self.model_time>max_time)[0][0]
            #    model_time_plot = self.model_time[:model_over_idx]
            #    model_roi0_plot = model_int[:model_over_idx]

            #ax1.plot(exp_time_plot,exp_roi0_plot,linestyle='',marker='o',markersize=4,color='black',label='Experiment')
            #ax1.plot(model_time_plot,model_roi0_plot,linestyle='-',color='red',label='Diffusion Model')
            ax1.plot(self.plot_exp_time,self.plot_exp_roi0,linestyle='',marker='o',markersize=4,color='black',label='Experiment')
            ax1.plot(self.cut_model_time,model_int[self.cut_model_idx],linestyle='-',color='red',label='Diffusion Model')
            ax1.set_xlabel('Time (s)')
            ax1.set_xlim(-5,1.1*np.min([np.max(self.cut_exp_time),np.max(self.cut_model_time)]))
            ax1.set_ylabel('Intensity (A.U.)')
            ax1.legend(loc=0)
            ax2.plot(self.cut_exp_time,model_res,marker='.',color='red',linestyle='')
            ax2.axhline(0,linestyle='--',color='black')
            ax2.set_ylabel('Residuals of Fit (A.U.)')
            plt.tight_layout()

            if gridpoint==False:
            #Would only be here if rank==0 or gridpoint==True.  If rank==0 and gridpoint==False, then this is the final model
                plt.savefig(self.output_prefix+"_Intensity_Timeseries.pdf",format='pdf')
            else:
                save_str = os.path.join(self.plot_dir,
                          "D_"+str(self.this_D)+"_f_"+str(self.this_mobile)+"_Intensity_Timeseries.pdf")
                plt.savefig(save_str,format='pdf')
            plt.close('all')

    def make_movie(self):
        file_prefix  = self.output_prefix + "_D_"+str(self.this_D)\
                     + "_mobile_" + str(self.this_mobile) + "_simulation_"\
                     + str(self.ensemble_idx)
        os.system("ffmpeg -f image2 -r 30 -i "+file_prefix+"_frame%04d.png -vcodec mpeg4 -y "+file_prefix+"_movie.mp4")
        for im_file in self.image_array:
            #os.system("rm "+im_file)
            os.remove(im_file)

    def print_intensity_timeseries(self):
        ofile = open(self.output_prefix+"_roi0_intensity_timeseries.csv",'w')
        ofile.write('Time(s),ExpIntensity(A.U.),ModelIntensity(A.U.),\n')
        for idx in range(len(self.exp_times)):
            if idx < self.norm_frames:
                ofile.write(str(self.exp_times[idx,0])+','+str(self.exp_data[idx,1])+',,\n')#,self.roi0_column])+',,\n')
            else:
                ofile.write(str(self.exp_times[idx,0])+','+str(self.exp_data[idx,1])+',')#,self.roi0_column])+',')
                try:
                    ofile.write(str(self.interpolated_model[idx-self.norm_frames])+",\n")
                except:
                    ofile.write(',\n')
        ofile.close()
        self.plot_roi_intensity_timeseries()             
    
    def report_best_model(self):
        ofile = open(self.output_prefix+"_best_5_models.csv",'w')
        ofile.write("#D(pix/step),D(um^2/s),F(ppt),r^2,RMSD(A.U.),\n")
        self.sorted_rmsd    = np.sort(self.rmsd_matrix,axis=None)
        top_5_rmsd  = self.sorted_rmsd[:5]
        for rmsd_idx in range(np.min([5,self.n_sets])):
            this_rmsd = top_5_rmsd[rmsd_idx]
            if self.n_d > 1 and self.n_m > 1:
                model_idx = np.where(self.rmsd_matrix==this_rmsd)
            
                rmsd_D_idx = model_idx[0][0]
                rmsd_m_idx = model_idx[1][0]
            elif self.n_d > 1:
                model_idx = np.where(self.rmsd_matrix==this_rmsd)
                rmsd_D_idx = model_idx[0][0]
                rmsd_m_idx = 0
            elif self.n_m > 1:
                model_idx = np.where(self.rmsd_matrix==this_rmsd)
                rmsd_D_idx = 0
                rmsd_m_idx = model_idx[0][0]
            else:
                rmsd_D_idx = 0
                rmsd_m_idx = 0
            rmsd_D = self.d_range[rmsd_D_idx]
            rmsd_m = self.m_range[rmsd_m_idx]

            rmsd_D_converted = 0.5 * rmsd_D**2 * self.scale**2 * (1./self.tstep)
            
            rounded_D = np.around(rmsd_D_converted,decimals=4)
            rounded_rmsd = np.around(this_rmsd,decimals=4)
            rounded_r2  = np.around(self.r2_matrix[rmsd_D_idx,rmsd_m_idx],decimals=4)
            ofile.write(str(rmsd_D)+","+str(rounded_D)+","+str(rmsd_m)+","\
                       +str(rounded_r2)+","+str(rounded_rmsd)+",\n")

            if rmsd_idx==0:
                print("\n\n\n==========================================")
                print("\tBest identified qFADD model is:")
                print("\t\tD = "+str(rounded_D)+" um^2/s ("+str(rmsd_D)+" pixels/step)")
                print("\t\tMobile Fraction = "+str(rmsd_m)+" parts per thousand")
                print("\t\tr^2  = "+str(rounded_r2))
                print("\t\tRMSD = "+str(rounded_rmsd)+" (A.U.)")
                print("==========================================")

        ofile.close()

    def plot_r2_map(self):
        if rank==0:
            if len(self.d_range) > 1 and len(self.m_range) > 1:
                plt.figure(figsize=(6.0,6.0))
                plt.contourf(
                    self.d_range_um, #self.d_range,
                    self.m_range,np.around(self.r2_matrix.T,decimals=3),
                    np.linspace(0.0,1.0,num=21),cmap='rainbow_r')
                plt.xlabel(r'D ($\rm{\mu m ^{2} / s}$)')
                plt.ylabel('Mobile Fraction (parts per thousand)')
                cb = plt.colorbar()
                cb.set_label(r'$\rm{r}^{2}$')
                plt.tight_layout()
                plt.savefig(self.output_prefix+"_r2_matrix.pdf",format='pdf')
                plt.close('all')
            elif len(self.d_range) > 1:
                plt.figure(figsize=(6.,4.))
                plt.plot(self.d_range_um,self.r2_matrix)
                plt.xlabel(r'D ($\rm{\mu m^{2}/s}$)')
                plt.ylabel(r'$\rm{r}^{2}$')
                plt.tight_layout()
                plt.savefig(self.output_prefix+"_r2_matrix.pdf",format='pdf')
                plt.close('all')
            elif len(self.m_range) > 1:
                plt.figure(figszie=(6.,4.))
                plt.plot(self.m_range,self.r2_matrix)
                plt.xlabel('Mobile Fraction (parts per thousand)')
                plt.ylabel(r'$\rm{r}^{2}$')
                plt.tight_layout()
                plt.savefig(self.output_prefix+'_r2_matrix.pdf',format='pdf')
                plt.close('all')
            else:
                print("Warning: Single (D_eff, mobile fraction) point"\
                     +" requested. No fit-quality plot created.")


    def plot_rmsd_map(self):
        if rank == 0:
            if len(self.d_range) > 1 and len(self.m_range) > 1:
                plt.figure(figsize=(6.0,6.0))
                plt.contourf(
                    self.d_range_um, #self.d_range,
                    self.m_range,np.around(self.rmsd_matrix.T,decimals=3),
                    np.linspace(0.0,1.0,num=21),cmap='rainbow')
                plt.xlabel(r'D ($\rm{\mu m ^{2} / s}$)')
                plt.ylabel('Mobile Fraction (parts per thousand)')
                cb = plt.colorbar()
                cb.set_label('RMSD (A.U.)')
                plt.tight_layout()
                plt.savefig(self.output_prefix+"_rmsd_matrix.pdf",format='pdf')
                plt.close('all')
            elif len(self.d_range) > 1:
                plt.figure(figsize=(6.,4.))
                plt.plot(self.d_range_um,self.rmsd_matrix)
                plt.xlabel(r'D ($\rm{\mu m ^{2} / s}$)')
                plt.ylabel('RMSD (A.U.)')
                plt.tight_layout()
                plt.savefig(self.output_prefix+"_rmsd_matrix.pdf",format='pdf')
                plt.close('all')
            elif len(self.m_range) > 1:
                plt.figure(figsize=(6.,4.))
                plt.plot(self.m_range,self.rmsd_matrix)
                plt.xlabel('Mobile Fraction (parts per thousand)')
                plt.ylabel('RMSD (A.U.)')
                plt.tight_layout()
                plt.savefig(self.output_prefix+"_rmsd_matrix.pdf",format='pdf')
                plt.close('all')

    def find_plateau(self):
        roi_column  = self.exp_data[:,1]#,self.roi0_column]
        max_intensity_point = np.where(roi_column==np.max(roi_column))[0][0]
        max_intensity_time  = self.exp_data[max_intensity_point,0]
        self.nsteps = int(np.ceil(max_intensity_time/self.tstep))
        pass
