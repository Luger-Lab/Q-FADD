#!/usr/bin/env python

import argparse
import os

def install_with_path_update(ifile,install_path):

    tempdat = open(ifile,'r')
    tempstr = tempdat.read()
    tempdat.close()

    outstr = tempstr.replace("INSTALL_PATH","\""+install_path+"\"")

    outdat = open(os.path.join(install_path,ifile),'w')
    outdat.write(outstr)
    outdat.close()

    os.system("chmod +x "+install_path+"/"+ifile)

    return

def install_no_path_update(ifile,install_path):
    os.system("cp "+ifile+" "+install_path+"/"+ifile)
    os.system("chmod +x "+install_path+"/"+ifile)
    return

parser = argparse.ArgumentParser()
parser.add_argument("-install_path",type=str,help='Path to desired installation point.',default='',dest='path')
parser.add_argument("--uninstall",action='store_true',dest='uninstall',help='Uninstall qFADD programs from -install_path')
args = parser.parse_args()

if args.path=='':
    print("\n\n\tERROR: You must provide an installation path!\n\n")
    quit()

if args.uninstall:
    FILELIST = ['qFADD.py','qFADD_library.py','image_analyzer.py','MovieAnalyzer_library.py','qFADD_gui.ui','image_analyzer_gui.ui','image_analyzer_gui.py','qFADD_gui.py']
    for FILE in FILELIST:
        os.remove(os.path.join(args.path,FILE))

#Install GUI programs, linking to proper .ui locations
install_with_path_update('image_analyzer_gui.py',args.path)
install_with_path_update('qFADD_gui.py',args.path)

#Copy the rest of the files to the install location
for fname in ['qFADD.py','qFADD_library.py','image_analyzer.py','MovieAnalyzer_library.py','qFADD_gui.ui','image_analyzer_gui.ui']:
    install_no_path_update(fname,args.path)
