#!/usr/bin/env python

''' Introductory Comment '''
import sys,os,math,pickle,glob
import itertools
import numpy as np
import subprocess
import getpass
from PySide2.QtWidgets import *  #QApplication,QMainWindow
from PySide2.QtCore import QFile,QObject
from PySide2 import QtCore,QtWidgets
from PySide2.QtUiTools import QUiLoader
from MovieAnalyzer_library import *
import bioformats,javabridge

install_path = "/home/sabo8529/lab_software/new_python_environments/qfadd_github/anaconda3/bin/"

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
        roi_buff = [int(self.roi_xbuff_line.text()),int(self.roi_ybuff_line.text())]
        analyzer = RecruitmentMovieAnalyzer()
        analyzer.SetParameters(protein_channel=self.prot_channel_line.text(),
                               nuclear_channel=self.nuc_channel_line.text(),
                               additional_rois=int(self.roi_spinbox.value()),
                               irrad_frame=int(self.irrad_frame_line.text()),
                               bleach_frames=int(self.irrad_frame_line.text()),
                               save_direct=self.out_direct_line.text(),
                               roi_buffer=roi_buff)
        if self.single_file_button.isChecked():
            analyzer.LoadFile(video_file=self.movie_file_line.text(),
                              roi_file=self.roi_file_line.text())
        else:
            analyzer.LoadDirectory(video_direct=self.batch_direct_line.text(),
                                   extension=self.movie_file_line.text(),
                                   roi_extension=self.roi_file_line.text())

        javabridge.start_vm(class_path=bioformats.JARS)
        t0 = time.time()
        #try:
        if 1:
            analyzer.ProcessFileList()
        #except:
        #    print("Error processing the filelist!!!")

        tf = time.time()
        print("Processed file(s) in "\
             +str(np.around((tf-t0)/60,decimals=3))\
             +" minutes.")
        javabridge.kill_vm()
        pass

    def MapIt(self,widget,name):
        return self.window.findChild(widget,name)

    def LoadUiOptions(self):
        
        #All the radio buttons
        self.batch_mode_button  = self.MapIt(QRadioButton,'BatchModeButton')
        self.single_file_button = self.MapIt(QRadioButton,'SingleFileButton')

        #All the push buttons
        self.batch_browse_button    = self.MapIt(QPushButton,'BatchDirectoryButton')
        self.movie_browse_button    = self.MapIt(QPushButton,'MovieBrowserButton')
        self.roi_browse_button      = self.MapIt(QPushButton,'ROIBrowserButton')
        self.outdir_browse_button   = self.MapIt(QPushButton,'OutputDirectoryButton')
        self.submit_button          = self.MapIt(QPushButton,'SubmitButton')
        self.save_options_button    = self.MapIt(QPushButton,'SaveButton')

        #Checkbox Buttons
        self.drift_checkbox     = self.MapIt(QCheckBox,'DriftCheckBox')
        self.bleach_correct_checkbox= self.MapIt(QCheckBox,'BleachCheckBox')

        #All the LineEdits
        self.batch_direct_line  = self.MapIt(QLineEdit,'BatchDirectoryLineEdit')
        self.movie_file_line    = self.MapIt(QLineEdit,'MovieFileLineEdit')
        self.roi_file_line      = self.MapIt(QLineEdit,'ROIFileLineEdit')
        self.out_direct_line    = self.MapIt(QLineEdit,'OutDirectLineEdit')
        self.nuc_channel_line   = self.MapIt(QLineEdit,'NucChannelLineEdit')
        self.prot_channel_line  = self.MapIt(QLineEdit,'ProtChannelLineEdit')
        self.roi_xbuff_line     = self.MapIt(QLineEdit,'ROIXBufferLineEdit')
        self.roi_ybuff_line     = self.MapIt(QLineEdit,'ROIYBufferLineEdit')
        self.irrad_frame_line   = self.MapIt(QLineEdit,'IrradFrameLineEdit')

        #Spinbox
        self.roi_spinbox        = self.MapIt(QSpinBox,'AdditionalRoiSpinbox')

        #A few labels can be changed based on checkmarking
        self.movie_label        = self.MapIt(QLabel,'MovieLabel')
        self.roi_label          = self.MapIt(QLabel,'ROILabel')

    def ChangeLabels(self):
        if self.batch_mode_button.isChecked():
            self.movie_label.setText('Movie File(s) Extension:')
            self.roi_label.setText('ROI File(s) Extension:')
        else:
            self.movie_label.setText('Movie File:')
            self.roi_label.setText('ROI File:')

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


    def SetDefaults(self):
        #set default I/O to Single File mode
        self.single_file_button.setChecked(True)

        self.user_home          = os.path.expanduser("~")
        self.preqfaddrc_path       = os.path.join(self.user_home,".preqfaddrc")
        self.out_direct_line.setText("./Movie_analysis")
        try:
            self.preqfaddrc    = pickle.load(open(self.preqfaddrc_path,'rb'))

            self.nuc_channel_line.setText(self.preqfaddrc.nuc_chan_text)
            self.prot_channel_line.setText(self.preqfaddrc.prot_chan_text)
            self.roi_xbuff_line.setText(self.preqfaddrc.roi_xbuff_text)
            self.roi_ybuff_line.setText(self.preqfaddrc.roi_ybuff_text)
            self.irrad_frame_line.setText(self.preqfaddrc.irrad_frame_text)
            self.roi_spinbox.setValue(self.preqfaddrc.roi_spinbox_value)

            if self.preqfaddrc.drift_correct:
                self.drift_checkbox.setChecked(QtCore.Qt.CheckState.Checked)
            else:
                self.drift_checkbox.setChecked(QtCore.Qt.CheckState.Unchecked)

            if self.preqfaddrc.bleach_correct:
                self.bleach_correct_checkbox.setChecked(QtCore.Qt.CheckState.Checked)
            else:
                self.bleach_correct_checkbox.setChecked(QtCore.Qt.CheckState.Unchecked)
        except:
            self.nuc_channel_line.setText('')
            self.prot_channel_line.setText('')
            self.roi_xbuff_line.setText('-10')
            self.roi_ybuff_line.setText('10')
            self.irrad_frame_line.setText('6')
            self.roi_spinbox.setValue(0)
            self.drift_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)
            self.bleach_correct_checkbox.setCheckState(QtCore.Qt.CheckState.Checked)

    def LinkUiOptions(self):
        self.submit_button.clicked.connect(self.SubmitAction)
        self.single_file_button.toggled.connect(self.FadeFileOptions)
        self.save_options_button.clicked.connect(self.SaveOptions)

        #File Browsers
        self.batch_browse_button.clicked.connect(self.OpenBatchDirectoryBrowser)
        self.movie_browse_button.clicked.connect(self.OpenMovieBrowser)
        self.roi_browse_button.clicked.connect(self.OpenROIBrowser)
        self.outdir_browse_button.clicked.connect(self.OpenOutDirectoryBrowser)

    def FadeFileOptions(self):
        self.ChangeLabels()
        if self.batch_mode_button.isChecked():
            self.UnFadeLineEdit(self.movie_file_line,'')
            self.UnFadeLineEdit(self.roi_file_line,'')
            try:
                self.UnFadeLineEdit(self.batch_direct_line,self.batch_direct_line_backup)
            except:
                self.UnFadeLineEdit(self.batch_direct_line,'')
        elif self.single_file_button.isChecked():
            self.UnFadeLineEdit(self.movie_file_line,'')
            self.UnFadeLineEdit(self.roi_file_line,'')
            self.batch_direct_line_backup = self.FadeLineEdit(self.batch_direct_line,'')

    def OpenMovieBrowser(self):
        self.file_window= FileBrowser(parent=self)
        if self.this_file != 'no file selected':
            self.movie_file_line.setText(self.this_file)

    def OpenROIBrowser(self):
        self.file_window= FileBrowser(parent=self)
        if self.this_file != 'no file selected':
            self.roi_file_line.setText(self.this_file)

    def OpenBatchDirectoryBrowser(self):
        self.file_window= DirectoryBrowser(parent=self)
        if self.this_direct != 'no directory selected':
            self.batch_direct_line.setText(self.this_direct)

    def OpenOutDirectoryBrowser(self):
        self.file_window= DirectoryBrowser(parent=self)
        if self.this_direct != 'no directory selected':
            self.out_direct_line.setText(self.this_direct)

    def SaveOptions(self):
        self.preqfaddrc_options = OptionsSaveObject()
        self.preqfaddrc_options.convert(self)
        self.preqfaddrc_options.save()

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
        self.preqfaddrc = parent.preqfaddrc_path

        #SLURM Options
        if parent.batch_mode_button.isChecked():
            self.batch_mode_button   = True
        else:
            self.batch_mode_button   = False

        self.nuc_chan_text      = parent.nuc_channel_line.text()
        self.prot_chan_text     = parent.prot_channel_line.text()
        self.roi_xbuff_text     = parent.roi_xbuff_line.text()
        self.roi_ybuff_text     = parent.roi_ybuff_line.text()
        self.irrad_frame_text   = parent.irrad_frame_line.text()
        self.roi_spinbox_value  = parent.roi_spinbox.value()

        if parent.drift_checkbox.checkState()==QtCore.Qt.CheckState.Unchecked:
            self.drift_correct = False
        else:
            self.drift_correct = True

        if parent.bleach_correct_checkbox.checkState()==QtCore.Qt.CheckState.Unchecked:
            self.bleach_correct = False
        else:
            self.bleach_correct = True

    def save(self):
        pickle.dump(self,open(self.preqfaddrc,'wb'))
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(os.path.join(install_path,"image_analyzer_gui.ui"))
    sys.exit(app.exec_())
