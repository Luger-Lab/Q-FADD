#!/usr/bin/env python

'''before you can run this script, run "pyside2-uic [ui file] > ui_mainwindow.py" '''

import sys,os,math,pickle,glob
import itertools
import numpy as np
import subprocess
import getpass
from PySide2.QtWidgets import *  #QApplication,QMainWindow
from PySide2.QtCore import QFile,QObject
from PySide2 import QtCore,QtWidgets
from PySide2.QtUiTools import QUiLoader

install_path = INSTALL_PATH

class MainWindow(QObject):
    def __init__(self,ui_file,parent=None):
        super(MainWindow,self).__init__(parent)
        ui_file = QFile(ui_file)
        ui_file.open(QFile.ReadOnly)

        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        
        self.LoadUiOptions()
        self.LinkUiOptions()
        self.SetDefaults()


        self.window.show()
    
    def SubmitAction(self):
        state = self.CheckResourceRequest()
        if state=='check_again':
            self.log_text_edit.appendPlainText('--Queue optimization complete: Press "Submit" again to submit the qFADD task(s)!')
        elif state=='ready':
            self.PrepareSLURM()
            if self.slurm_button.isChecked():
                self.SubmitSLURM()
                self.log_text_edit.appendPlainText("--qFADD job(s) have been submitted to the queue!")
            elif self.localhost_button.isChecked():
                self.RunLocal()
        else:
            self.log_text_edit.appendPlainText('ERROR: "state" variable somehow set to non-standard value???')

    def PrepareSLURM(self):
        self.slurm_list = []
        if self.batch_mode_button.isChecked():
            self.BatchPrepare()
        else:
            if self.localhost_button.isChecked():
                self.PrepareSingleLocalhost(self.output_prefix_line.text())
            elif self.slurm_button.isChecked():
                self.PrepareSingleSLURM(self.output_prefix_line.text())
                self.log_text_edit.appendPlainText("--Queued Q-FADD analysis on single nucleus "+str(maskfile)+".")

    def SubmitSLURM(self):
        for sfile in self.slurm_list:
            this_process= subprocess.Popen(["sbatch",sfile],stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    def RunLocal(self):
        #this_process    = subprocess.Popen(["nohup","bash "+self.local_run_filename],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        os.system("nohup bash "+self.local_run_filename+" &> "+self.local_run_filename+".log &")

    def BatchPrepare(self):
        base_directory  = self.batch_dir_line.text()
        mask_pattern    = self.mask_file_line.text()
        roi_pattern     = self.roi_file_line.text()
        kinetics_pattern= self.kinetics_file_line.text()

        mask_list   = glob.glob(os.path.join(base_directory,"*"+mask_pattern))

        if self.localhost_button.isChecked():
            self.local_run_filename = self.output_prefix_line.text()+'_jobs.sh'
            self.local_run_file = open(self.local_run_filename,'w')
            self.log_text_edit.appendPlainText("--qFADD tasks being stored in \""\
                                              + self.local_run_filename+"\"")
            
        for maskfile in mask_list:
            self.log_text_edit.appendPlainText("--Maskfile: "+maskfile)
            prefix  = os.path.basename(maskfile).replace(mask_pattern,'')
            roifile = os.path.join(base_directory,prefix+roi_pattern)
            kfile   = os.path.join(base_directory,prefix+kinetics_pattern)

            roi_exists  = os.path.isfile(roifile)
            k_exists    = os.path.isfile(kfile)

            out_direct  = os.path.join(prefix+mask_pattern.split('.')[0])

            runtime_log = os.path.join(out_direct,
                             self.output_prefix_line.text()+".log")

            if not os.path.isdir(out_direct):
                try:
                    os.mkdir(out_direct)
                except:
                    pass


            if roi_exists and k_exists:
                if self.slurm_button.isChecked():
                    self.PrepareSingleSLURM(prefix,maskf=maskfile,roif=roifile,kf=kfile)
                elif self.localhost_button.isChecked():
                    self.log_text_edit.appendPlainText("--Added nucleus "+str(maskfile)+" to analysis list.")
                    self.local_run_file.write("mpirun -np "+self.num_procs_line.text()+" qFADD.py")
                    self.local_run_file.write(" \\\n\t -mf "+maskfile)
                    self.local_run_file.write(" \\\n\t -roif "+roifile)
                    self.local_run_file.write(" \\\n\t -kf "+kfile)
                    self.local_run_file.write(" \\\n\t -roic "+str(self.roic_spinbox.value()))
                    self.local_run_file.write(" \\\n\t -out "+self.output_prefix_line.text())
                    self.local_run_file.write(" \\\n\t -t0 "+self.offset_time_line.text())
                    self.local_run_file.write(" \\\n\t -norm "+self.norm_frames_line.text())
                    self.local_run_file.write(" \\\n\t -nmol "+self.num_molecules_line.text())
                    self.local_run_file.write(" \\\n\t -mobile_min "+self.min_mobile_line.text())
                    self.local_run_file.write(" \\\n\t -mobile_max "+self.max_mobile_line.text())
                    self.local_run_file.write(" \\\n\t -mobile_step "+self.mobile_step_line.text())
                    self.local_run_file.write(" \\\n\t -pix "+self.pixel_res_line.text())
                    self.local_run_file.write(" \\\n\t -Dmin "+self.min_D_line.text())
                    self.local_run_file.write(" \\\n\t -Dmax "+self.max_D_line.text())
                    self.local_run_file.write(" \\\n\t -Dstep "+self.D_step_line.text())
                    self.local_run_file.write(" \\\n\t -ensemble_size "+self.ensemble_line.text())
                    self.local_run_file.write(" \\\n\t -nsteps "+self.nsteps_line.text())
                    self.local_run_file.write(" \\\n\t -tstep "+self.timestep_line.text())
                    if self.exp_length_button.isChecked():
                        self.local_run_file.write(" \\\n\t --find_plat")
                    if self.model_select_dropdown.currentText()=='Average of Models':
                        self.local_run_file.write(" \\\n\t --use_avg")
                    if self.plot_all_checkbox.checkState()==QtCore.Qt.CheckState.Checked:
                        self.local_run_file.write(" \\\n\t --plot_all")
                    self.local_run_file.write(" \\\n\t &> "+runtime_log+"\n\n")
                self.log_text_edit.appendPlainText("--Added nucleus "+str(maskfile)+" to the analysis list.")
            else:
                if not roi_exists:
                    self.log_text_edit.appendPlainText("--WARNING: No ROI File ("+roifile+") found to be associated with Mask File ("+maskfile+").  Excluding "+maskfile+" from analysis!!!")
                if not k_exists:
                    self.log_text_edit.appendPlainText("--WARNING: No Kinetics File ("+kfile+") found to be associated with the Kinetics File ("+maskfile+").  Excluding "+maskfile+" from analysis!!!")
        
        if self.localhost_button.isChecked():
            self.local_run_file.close()

    def PrepareSingleLocalhost(self,file_prefix,maskf='',roif='',kf=''):
        if maskf=='':
            maskf   = self.mask_file_line.text()

        output_dir = os.path.splitext(os.path.basename(maskf))[0]
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir)
            except:
                pass


        self.local_run_filename = self.PrepareSingleSLURM(file_prefix,maskf=maskf,roif=roif,kf=kf)
        self.log_text_edit.appendPlainText("--Prepared local Q-FADD analysis on single nucleus "+str(maskf)+".")

    def PrepareSingleSLURM(self,file_prefix,maskf='',roif='',kf=''):
        if maskf=='':
            maskf   = self.mask_file_line.text()
        if roif=='':
            roif    = self.roi_file_line.text()
        if kf=='':
            kf      = self.kinetics_file_line.text()
        
        output_dir = os.path.splitext(os.path.basename(maskf))[0]
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir)
            except:
                pass

        log_file = os.path.join(output_dir,
                      self.output_prefix_line.text())

        slurm_file = open(file_prefix+".slurm",'w')
        slurm_file.write("#!/bin/bash\n\n")
        slurm_file.write("#SBATCH --job-name="+file_prefix+"\n")
        slurm_file.write("#SBATCH --nodes="+self.num_nodes_line.text()+"\n")
        slurm_file.write("#SBATCH --ntasks-per-node="+self.num_procs_line.text()+"\n")
        slurm_file.write("#SBATCH --walltime="+self.walltime_line.text()+"\n")
        slurm_file.write("#SBATCH --partition="+self.partition_line.text()+"\n")
        try:
            if self.email_line.text().replace(' ','') != '':
                slurm_file.write("#SBATCH --email="+self.email_line.text().replace(' ','')+"\n\n")
            else:
                slurm_file.write("\n")
        except:
            slurm_file.write("\n")
        
        if self.slurm_button.isChecked():
            slurm_file.write("mpirun qFADD.py")
        else:
            slurm_file.write("mpirun -np "+self.num_procs_line.text()+" qFADD.py")
        
        
        slurm_file.write(" \\\n\t -mf "+maskf)
        slurm_file.write(" \\\n\t -roif "+roif)
        slurm_file.write(" \\\n\t -kf "+kf)
        slurm_file.write(" \\\n\t -roic "+str(self.roic_spinbox.value()))
        slurm_file.write(" \\\n\t -out "+self.output_prefix_line.text())
        slurm_file.write(" \\\n\t -t0 "+self.offset_time_line.text())
        slurm_file.write(" \\\n\t -norm "+self.norm_frames_line.text())
        slurm_file.write(" \\\n\t -nmol "+self.num_molecules_line.text())
        slurm_file.write(" \\\n\t -mobile_min "+self.min_mobile_line.text())
        slurm_file.write(" \\\n\t -mobile_max "+self.max_mobile_line.text())
        slurm_file.write(" \\\n\t -mobile_step "+self.mobile_step_line.text())
        slurm_file.write(" \\\n\t -pix "+self.pixel_res_line.text())
        slurm_file.write(" \\\n\t -Dmin "+self.min_D_line.text())
        slurm_file.write(" \\\n\t -Dmax "+self.max_D_line.text())
        slurm_file.write(" \\\n\t -Dstep "+self.D_step_line.text())
        slurm_file.write(" \\\n\t -ensemble_size "+self.ensemble_line.text())
        slurm_file.write(" \\\n\t -nsteps "+self.nsteps_line.text())
        slurm_file.write(" \\\n\t -tstep "+self.timestep_line.text())
        if self.exp_length_button.isChecked():
            slurm_file.write(" \\\n\t --find_plat")
        if self.model_select_dropdown.currentText()=='Average of Models':
            slurm_file.write(" \\\n\t --use_avg")
        if self.plot_all_checkbox.checkState()==QtCore.Qt.CheckState.Checked:
            slurm_file.write(" \\\n\t --plot_all")
        if self.slurm_button.isChecked():
            slurm_file.write("\n")
        else:
            if maskf=='':
                slurm_file.write(" \\\n\t &> "+log_file+"\n")

        slurm_file.close()
        self.slurm_list.append(file_prefix+".slurm")

        return file_prefix+'.slurm'

    def ClearCache(self):
        self.slurm_list = []
        self.log_text_edit.appendPlainText("--Job Submission Queue has been cleared.")
        self.CheckJobs()
        pass

    def CheckJobs(self):
        if self.slurm_button.isChecked():
            self.log_text_edit.appendPlainText("--Gathering data for currently running jobs.")
            this_process= subprocess.Popen(["squeue"," -u "+getpass.getuser()+" > current_jobs\""],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            process_2   = subprocess.Popen(["gedit", "current_jobs"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        else:
            self.log_text_edit.appendPlainText("--To view job statuses, run \"top -u "+getpass.getuser()+"\" in a separate terminal window.")
        

    def CheckResourceRequest(self):
        nodes   = float(self.num_nodes_line.displayText())
        procs   = float(self.num_procs_line.displayText())
        dmin    = float(self.min_D_line.displayText())
        dmax    = float(self.max_D_line.displayText())
        dstep   = float(self.D_step_line.displayText())
        fmin    = float(self.min_mobile_line.displayText())
        fmax    = float(self.max_mobile_line.displayText())
        fstep   = float(self.mobile_step_line.displayText())

        d_points= np.arange(dmin,dmax+1,dstep)
        f_points= np.arange(fmin,fmax+1,fstep)
        Nd      = len(d_points)
        Nf      = len(f_points)
        n_combos= Nd*Nf

        self.n_combos   = n_combos
        total_procs     = nodes * procs

        ratio = n_combos / procs #Determine what optimum node count would be for this number of combos
        self.ratio  = ratio

        if 0:
        #if total_procs > n_combos and not self.automate_checkbox.isChecked():
            self.log_text_edit.appendPlainText("--Requested number of processors ("+str(total_procs)+") exceeds the total number of grid points ("+str(Nd)+"x"+str(Nf)+"="+str(n_combos)+").  Optimizing number of requested nodes accordingly.")
            diff = total_procs - n_combos
            node_reduce = math.floor(diff/procs)
            nodes -= node_reduce
            self.num_nodes_line.setText(str(int(nodes)))
            if int(nodes)==1:
                if procs > n_combos:
                    self.log_text_edit.appendPlainText("--Number of combos still too few for requested number of procs per node, reducing requested number of processors accordingly.")
                    self.num_procs_line.setText(str(n_combos))

        if self.automate_checkbox.isChecked():
            self.OptimizeSUs()
            self.automate_checkbox.setChecked(False)
            return "check_again"
        else:
            return "ready"           

    def CountLoops(self,size,n_proc,n_nodes):
        ratio = size/(n_proc*n_nodes)
        n_loops = math.ceil(ratio)
        final_loop_cpu  = (ratio % 1) * (n_proc * n_nodes)
        if final_loop_cpu == 0:
            return n_loops, 0.0
        else:
            wasted_nodes    = ((n_proc * n_nodes) - final_loop_cpu)/n_proc

            return n_loops, wasted_nodes

    def OptimizeSUs(self):
        max_proc_per_node   = float(self.num_procs_line.text())
        max_nodes           = float(self.num_nodes_line.text())
        
        proc_range = np.arange(1,max_proc_per_node+1,dtype=int)
        node_range = np.arange(1,max_nodes+1,dtype=int)
        combos          = list(itertools.product(proc_range,node_range))
        n_combos   = len(combos)
        self.log_text_edit.appendPlainText("--There are "+str(self.n_combos)+" grid points.")
        best_eff        = 9999999999999999999
        best_nodes      = 0
        best_procs      = 0
        best_n_loops    = 99999999999

        for idx in range(n_combos):
            this_proc   = combos[idx][0]
            this_node   = combos[idx][1]
            this_n_loops, this_waste= self.CountLoops(self.n_combos,this_proc,this_node)
            if ((this_waste+this_n_loops) < best_eff) and (this_n_loops < best_n_loops):
                best_eff = this_waste + (this_n_loops)
                best_proc_count = this_proc
                best_node_count = this_node
                best_n_loops    = this_n_loops
        self.num_procs_line.setText(str(best_proc_count))
        self.num_nodes_line.setText(str(best_node_count))
        

    def MapIt(self,widget,name):
        return self.window.findChild(widget,name)

    def LoadUiOptions(self):
        
        #All the radio buttons
        self.batch_mode_button  = self.MapIt(QRadioButton,'BatchModeButton')
        self.single_file_button = self.MapIt(QRadioButton,'SingleFileButton')
        self.exp_length_button  = self.MapIt(QRadioButton,'ExpLengthButton')
        self.fixed_length_button= self.MapIt(QRadioButton,'FixedLengthButton')
        self.localhost_button   = self.MapIt(QRadioButton,'LocalHostButton')
        self.slurm_button       = self.MapIt(QRadioButton,'SLURMButton')
        self.avg_sampled_button = self.MapIt(QRadioButton,'AvgSampledButton')

        #All the push buttons
        self.mask_browse_button = self.MapIt(QPushButton,'MaskFileButton')
        self.batch_browse_button= self.MapIt(QPushButton,'BatchDirectoryButton')
        self.roi_browse_button  = self.MapIt(QPushButton,'ROIFileButton')
        self.kfile_browse_button= self.MapIt(QPushButton,'KineticsFileButton')
        self.save_slurm_button  = self.MapIt(QPushButton,'SlurmSettingsSaveButton')
        self.clear_cache_button = self.MapIt(QPushButton,'ClearCacheButton')
        self.submit_button      = self.MapIt(QPushButton,'SubmitButton')

        #Mediod or Average model selection
        self.model_select_dropdown = self.MapIt(QComboBox,'ModelSelectDropdown')
        self.model_select_dropdown.addItem('Median Model')
        self.model_select_dropdown.addItem('Average of Models')

        #Checkbox Buttons
        self.automate_checkbox  = self.MapIt(QCheckBox,'AutomateSUCheckBox')
        self.plot_all_checkbox  = self.MapIt(QCheckBox,'PlotAllCheckBox')

        #Roi Column Selection Spinbox
        self.roic_spinbox       = self.MapIt(QSpinBox,'ROIColumnSpinBox')

        #All the LineEdits
        self.offset_time_line   = self.MapIt(QLineEdit,'OffsetTimeLineEdit')
        self.norm_frames_line   = self.MapIt(QLineEdit,'NormFramesLineEdit')
        self.num_molecules_line = self.MapIt(QLineEdit,'NumMoleculesLineEdit')
        self.pixel_res_line     = self.MapIt(QLineEdit,'PixelResolutionLineEdit')
        self.timestep_line      = self.MapIt(QLineEdit,'TimestepLineEdit')
        self.min_mobile_line    = self.MapIt(QLineEdit,'MinFLineEdit')
        self.max_mobile_line    = self.MapIt(QLineEdit,'MaxFLineEdit')
        self.mobile_step_line   = self.MapIt(QLineEdit,'FStepLineEdit')
        self.min_D_line         = self.MapIt(QLineEdit,'MinDLineEdit')
        self.max_D_line         = self.MapIt(QLineEdit,'MaxDLineEdit')
        self.D_step_line        = self.MapIt(QLineEdit,'DStepLineEdit')
        self.ensemble_line      = self.MapIt(QLineEdit,'EnsembleLineEdit')
        self.nsteps_line        = self.MapIt(QLineEdit,'NStepsLineEdit')
        self.output_prefix_line = self.MapIt(QLineEdit,'PrefixLineEdit')
        self.batch_dir_line     = self.MapIt(QLineEdit,'BatchDirectoryLineEdit')
        self.mask_file_line     = self.MapIt(QLineEdit,'MaskFileLineEdit')
        self.roi_file_line      = self.MapIt(QLineEdit,'ROIFileLineEdit')
        self.kinetics_file_line = self.MapIt(QLineEdit,'KineticsFileLineEdit')
        self.num_nodes_line     = self.MapIt(QLineEdit,'NumNodeLineEdit')
        self.num_procs_line     = self.MapIt(QLineEdit,'NumProcLineEdit')
        self.walltime_line      = self.MapIt(QLineEdit,'WalltimeLineEdit')
        self.partition_line     = self.MapIt(QLineEdit,'PartitionNameLineEdit')
        self.email_line         = self.MapIt(QLineEdit,'EmailLineEdit')

        #A few labels can be changed based on checkmarking
        self.node_label         = self.MapIt(QLabel,'NodesLabel')
        self.processors_label   = self.MapIt(QLabel,'ProcessorsLabel')
        self.mask_file_label    = self.MapIt(QLabel,'MaskLabel')
        self.roi_file_label     = self.MapIt(QLabel,'ROILabel')
        self.kfile_label        = self.MapIt(QLabel,'KineticsLabel')

        #D value labels, updated in real-time with inputs
        self.d_min_label        = self.MapIt(QLabel,'d_min_label')
        self.d_max_label        = self.MapIt(QLabel,'d_max_label')
        self.d_step_label       = self.MapIt(QLabel,'d_step_label')

        #The Log Report Window
        self.log_text_edit      = self.MapIt(QPlainTextEdit,'LogTextEdit')
        self.log_text_edit.setReadOnly(True)

    def ChangeLabels(self):
        if self.automate_checkbox.isChecked():
            self.node_label.setText('Maximum Nodes:')
            self.processors_label.setText('Max CPUs per Node:')
        else:
            self.node_label.setText('Number of Nodes:')
            self.processors_label.setText('CPUs Per Node:')

        if self.batch_mode_button.isChecked():
            self.mask_file_label.setText('Mask Extension:')
            self.roi_file_label.setText('ROI Extension:')
            self.kfile_label.setText('Intensity Kinetics Extension:')
        else:
            self.mask_file_label.setText('Nuclear Mask File:')
            self.roi_file_label.setText('ROI Boundary File:')
            self.kfile_label.setText('Intensity Kinetics File:')

    def FadeLineEdit(self,LineEdit,text):
        backup_value = LineEdit.text()
        LineEdit.setText(text)
        LineEdit.setStyleSheet('''QLineEdit {background-color: rgb(144,144,144)}''')
        LineEdit.setReadOnly(True)
        return backup_value

    def UnFadeLineEdit(self,LineEdit,text):
        LineEdit.setText(text)
        LineEdit.setReadOnly(False)
        LineEdit.setStyleSheet('''QLineEdit {background-color: rgb(255,255,255)}''')


    def FadeSLURMOptions(self):
        if self.localhost_button.isChecked():
            self.num_nodes_line_backup  = self.FadeLineEdit(self.num_nodes_line,'1')
            self.walltime_line_backup   = self.FadeLineEdit(self.walltime_line,'')
            self.partition_line_backup  = self.FadeLineEdit(self.partition_line,'')
            self.email_line_backup      = self.FadeLineEdit(self.email_line,'')
            self.automate_checkbox.setChecked(False)
            self.automate_checkbox.setEnabled(False)
        elif self.slurm_button.isChecked():
            self.automate_checkbox.setEnabled(True)
            self.UnFadeLineEdit(self.num_nodes_line,self.num_nodes_line_backup)
            self.UnFadeLineEdit(self.walltime_line,self.walltime_line_backup)
            self.UnFadeLineEdit(self.partition_line,self.partition_line_backup)
            self.UnFadeLineEdit(self.email_line,self.email_line_backup)

    def FadeFileOptions(self):
        self.ChangeLabels()
        if self.batch_mode_button.isChecked():
            #self.mask_file_line_backup      = self.FadeLineEdit(self.mask_file_line,'')
            #self.roi_file_line_backup       = self.FadeLineEdit(self.roi_file_line,'')
            #self.kinetics_file_line_backup  = self.FadeLineEdit(self.kinetics_file_line,'')
            self.UnFadeLineEdit(self.mask_file_line,'')
            self.UnFadeLineEdit(self.roi_file_line,'')
            self.UnFadeLineEdit(self.kinetics_file_line,'')
            
            try:
                self.UnFadeLineEdit(self.batch_dir_line,self.batch_dir_line_backup)
            except:
                self.UnFadeLineEdit(self.batch_dir_line,'')
        elif self.single_file_button.isChecked():
            self.UnFadeLineEdit(self.mask_file_line,'')
            self.UnFadeLineEdit(self.roi_file_line,'')
            self.UnFadeLineEdit(self.kinetics_file_line,'')
            self.batch_dir_line_backup  = self.FadeLineEdit(self.batch_dir_line,'')

    def SetDefaults(self):
        #set default I/O to Single File mode
        self.single_file_button.setChecked(True)

        #Slurm vs localhost - pull defaults from ~/.qfaddrc file
        self.user_home          = os.path.expanduser("~")
        self.qfaddrc_path       = os.path.join(self.user_home,".qfaddrc")
        self.output_prefix_line.setText("qFADD_analysis")
        try:
            self.qfaddrc    = pickle.load(open(self.qfaddrc_path,'rb'))

            #Submission Options
            self.localhost_button.setChecked(self.qfaddrc.localhost_button)
            self.slurm_button.setChecked(self.qfaddrc.slurm_button)
            self.num_nodes_line.setText(self.qfaddrc.num_nodes_text)
            self.num_procs_line.setText(self.qfaddrc.num_procs_text)
            self.walltime_line.setText(self.qfaddrc.walltime_text)
            self.partition_line.setText(self.qfaddrc.partition_text)
            self.automate_checkbox.setChecked(self.qfaddrc.automate_SUs)
            self.email_line.setText(self.qfaddrc.email_text)

            #qFADD Parameters
            self.offset_time_line.setText(self.qfaddrc.offset_time_text)
            self.norm_frames_line.setText(self.qfaddrc.norm_frames_text)
            self.num_molecules_line.setText(self.qfaddrc.num_molecules_text)
            self.pixel_res_line.setText(self.qfaddrc.pixel_res_text)
            self.timestep_line.setText(self.qfaddrc.timestep_text)
            self.min_mobile_line.setText(self.qfaddrc.min_mobile_text)
            self.max_mobile_line.setText(self.qfaddrc.max_mobile_text)
            self.mobile_step_line.setText(self.qfaddrc.mobile_step_text)
            self.min_D_line.setText(self.qfaddrc.min_D_text)
            self.max_D_line.setText(self.qfaddrc.max_D_text)
            self.D_step_line.setText(self.qfaddrc.D_step_text)
            self.nsteps_line.setText(self.qfaddrc.nsteps_text)
            self.ensemble_line.setText(self.qfaddrc.ensemble_text)
            if self.qfaddrc.fixed_length_button:
                self.fixed_length_button.setChecked(True)
            else:
                self.exp_length_button.setChecked(True)

            if self.qfaddrc.plot_all_checked:
                self.plot_all_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                self.plot_all_checkbox.setCheckState(QtCore.Qt.CheckState.Unchecked)

            self.log_text_edit.appendPlainText("--Loaded defaults from "+self.qfaddrc_path)
        except:
            self.log_text_edit.appendPlainText("--Could not load .qfaddrc file ("+self.qfaddrc_path+"), setting defaults internally")

            #Submission Options
            self.localhost_button.setChecked(False)
            self.slurm_button.setChecked(True)
            self.num_nodes_line.setText("1")
            self.num_procs_line.setText("24")
            self.walltime_line.setText("48:00:00")
            self.partition_line.setText("normal")
            self.automate_checkbox.setChecked(False)

            #qFADD parameters
            self.offset_time_line.setText("11")
            self.norm_frames_line.setText("6")
            self.num_molecules_line.setText("10000")
            self.pixel_res_line.setText("0.08677")
            self.timestep_line.setText("0.1824")
            self.min_mobile_line.setText("400")
            self.max_mobile_line.setText("800")
            self.mobile_step_line.setText("50")
            self.min_D_line.setText("4")
            self.max_D_line.setText("15")
            self.D_step_line.setText("1")
            self.nsteps_line.setText("350")
            self.fixed_length_button.setChecked(True)
            self.ensemble_line.setText("11")
            self.plot_all_checkbox.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def LinkUiOptions(self):
        self.save_slurm_button.clicked.connect(self.SaveOptions)
        self.submit_button.clicked.connect(self.SubmitAction)
        self.localhost_button.toggled.connect(self.FadeSLURMOptions)
        self.single_file_button.toggled.connect(self.FadeFileOptions)
        self.automate_checkbox.stateChanged.connect(self.ChangeLabels)
        self.clear_cache_button.clicked.connect(self.ClearCache)

        #File Browsers
        self.mask_browse_button.clicked.connect(self.OpenMaskBrowser)
        self.roi_browse_button.clicked.connect(self.OpenROIBrowser)
        self.kfile_browse_button.clicked.connect(self.OpenKineticsBrowser)
        self.batch_browse_button.clicked.connect(self.OpenBatchDirectoryBrowser)

        #Converting D value labels
        self.min_D_line.textChanged.connect(self.min_D_line_changed)
        self.max_D_line.textChanged.connect(self.max_D_line_changed)
        self.D_step_line.textChanged.connect(self.D_step_line_changed)

    def min_D_line_changed(self, input_str):
        self.d_label_convert(input_str,'min')

    def max_D_line_changed(self, input_str):
        self.d_label_convert(input_str,'max')

    def D_step_line_changed(self, input_str):
        self.d_label_convert(input_str,'step')

    def d_label_convert(self,input_str,which_label):
        try:
            value = int(input_str.replace(' ',''))
            pix_value = float(self.pixel_res_line.text().replace(' ',''))
            tstep   = float(self.timestep_line.text().replace(' ',''))

            conv_value = 0.5 * value**2 * pix_value**2 * (1./tstep)
            conv_value = str(np.around(conv_value,decimals=3))
            label = "pix/step ("+conv_value+" &mu;m<sup>2</sup>/s)"

            if which_label=='min':
                self.d_min_label.setText(label)
            elif which_label=='max':
                self.d_max_label.setText(label)
            elif which_label=='step':
                self.d_step_label.setText(label)
        except:
            if which_label=='min':
                self.d_min_label.setText("pix/step")
            elif which_label=='max':
                self.d_max_label.setText("pix/step")
            elif which_label=='step':
                self.d_step_label.setText("pix/step")

    def OpenMaskBrowser(self):
        self.file_window= FileBrowser(parent=self)
        if self.this_file != 'no file selected':
            self.mask_file_line.setText(self.this_file)
        #self.file_window.show()

    def OpenROIBrowser(self):
        self.file_window= FileBrowser(parent=self)
        if self.this_file != 'no file selected':
            self.roi_file_line.setText(self.this_file)

    def OpenKineticsBrowser(self):
        self.file_window= FileBrowser(parent=self)
        if self.this_file != 'no file selected':
            self.kinetics_file_line.setText(self.this_file)

    def OpenBatchDirectoryBrowser(self):
        self.file_window= DirectoryBrowser(parent=self)
        if self.this_direct != 'no directory selected':
            self.batch_dir_line.setText(self.this_direct)

    def SaveOptions(self):
        self.qfaddrc_options = OptionsSaveObject()
        self.qfaddrc_options.convert(self)
        self.qfaddrc_options.save()

class FileBrowser(QMainWindow):
    def __init__(self,parent=None):
        super(FileBrowser,self).__init__()
        self.title  = 'File Browser Window'
        self.setWindowTitle(self.title)
        self.openFileNameDialog(parent=parent)

    def openFileNameDialog(self,parent=None):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"File Browser", "","All Files (*)", options=options)
        if fileName:
            parent.this_file = fileName
        else:
            parent.this_file = "no file selected"

class DirectoryBrowser(QMainWindow):
    def __init__(self,parent=None):
        super(DirectoryBrowser,self).__init__()
        self.setWindowTitle('Directory Browser')
        self.openDirectoryDialog(parent=parent)

    def openDirectoryDialog(self,parent=None):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        directName = dialog.getExistingDirectory(self,'Directory Browser')
        if directName:
            parent.this_direct = directName
        else:
            parent.this_direct = 'no directory selected'



class OptionsSaveObject(object):
    def convert(self,parent):
        self.qfaddrc = parent.qfaddrc_path

        #SLURM Options
        if parent.localhost_button.isChecked():
            self.localhost_button   = True
        else:
            self.localhost_button   = False

        if not self.localhost_button:
            self.slurm_button   = True
        else:
            self.slurm_button   = False

        self.num_nodes_text     = parent.num_nodes_line.text()
        self.num_procs_text     = parent.num_procs_line.text()
        self.walltime_text      = parent.walltime_line.text()
        self.partition_text     = parent.partition_line.text()
        self.automate_SUs       = parent.automate_checkbox.isChecked()
        self.email_text         = parent.email_line.text()

        #qFADD Parameters
        self.offset_time_text   = parent.offset_time_line.text()
        self.norm_frames_text   = parent.norm_frames_line.text()
        self.num_molecules_text = parent.num_molecules_line.text()
        self.pixel_res_text     = parent.pixel_res_line.text()
        self.timestep_text      = parent.timestep_line.text()
        self.min_mobile_text    = parent.min_mobile_line.text()
        self.max_mobile_text    = parent.max_mobile_line.text()
        self.mobile_step_text   = parent.mobile_step_line.text()
        self.min_D_text         = parent.min_D_line.text()
        self.max_D_text         = parent.max_D_line.text()
        self.D_step_text        = parent.D_step_line.text()
        self.nsteps_text        = parent.nsteps_line.text()
        self.ensemble_text      = parent.ensemble_line.text()
        
        if parent.fixed_length_button.isChecked():
            self.fixed_length_button = True
        else:
            self.fixed_length_button = False

        if parent.plot_all_checkbox.checkState()==QtCore.Qt.CheckState.Unchecked:
            self.plot_all_checked = False
        else:
            self.plot_all_checked = True

    def save(self):
        pickle.dump(self,open(self.qfaddrc,'wb'))
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(os.path.join(install_path,"qFADD_gui.ui"))
    sys.exit(app.exec_())
