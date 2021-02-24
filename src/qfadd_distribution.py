#!/usr/bin/env python

import numpy as np
import argparse
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from scipy.stats import t

#define p-value function
def ttest(avg1,sem1,n1,avg2,sem2,n2):
    sed = np.sqrt(np.power(sem1,2) + np.power(sem2,2))
    t_stat = np.divide(np.subtract(avg1,avg2),sed)
    df = n1+n2 - 2
    p = (1.0 - t.cdf(np.abs(t_stat),df))*2.0
    return p

#Build input parser
parser = argparse.ArgumentParser()
parser.add_argument("-flist",help='Space delimited list of text files, each listing files to combine per population',type=str,default='')
parser.add_argument("-dlist",help='Space delimited list of distribution files for comparing different populations of Q-FADD results (Ex: "-dlist PARP1.txt PARP2.txt HPF1.txt")',type=str,default='')
parser.add_argument("-labels",help='Labels for violin plot populations, space delimited. (Ex: "-labels PARP1 PARP2 HPF1")',type=str,default='')
parser.add_argument("-o",help='Output prefix for violin plot and text file generation',type=str,default='qfadd_distribution')
args = parser.parse_args()

#Determine if analyzing list of QFADD outputs or list of QFADD distributions
if args.flist!='':
    inlist = args.flist.split(" ")
    if args.labels != '':
        labels = args.labels.split(" ")
    else:
        labels = inlist
else:
    print('ERROR: You must provide at least one filelist!')
    quit()

#If list of qFADD results is provided, build a distribution of D and F values
for idx in range(len(inlist)):
    filelist = np.genfromtxt(inlist[idx],dtype=str)
    d_list = np.array([],dtype=float)
    f_list = np.array([],dtype=float)
    if args.labels != '':
        write_files = True
        ofile = open(args.o+"_"+labels[idx]+"_best_fit_models.dat",'w')
        ofile.write("#model file, D (um^2/s), Mobile Fraction (ppt),r^2\n")
    else:
        write_files = False
    for FILE in filelist:
        d,dconv,f,r2,rmsd,dum = np.genfromtxt(FILE,delimiter=',',dtype=float,unpack=True)
        d_list = np.append(d_list,dconv[0])
        f_list = np.append(f_list,f[0])
        if write_files:
            ofile.write(FILE+","+str(dconv[0])+","+str(f[0])+","+str(r2[0])+"\n")
    print("Cumulative Q-FADD results stored in "+args.o+"_best_fit_models.dat")
    if write_files:
        ofile.close()

    plt.figure(figsize=(3.,3.))
    ax1 = plt.subplot(111)
    ax1.violinplot(d_list,[0],showmeans=True)
    plt.ylabel(r'$\rm{D_{eff} (\mu m^{2}/s)}$')
    plt.xticks([],[])
    plt.tight_layout()
    plt.savefig(args.o+"_Deff_Violin.pdf",format='pdf')
    plt.close('all')

    plt.figure(figsize=(3.,3.))
    ax1 = plt.subplot(111)
    ax1.violinplot(f_list,[0],showmeans=True)
    plt.ylabel('Mobile Fraction (ppt)')
    plt.xticks([],[])
    plt.tight_layout()
    plt.savefig(args.o+"_MobileFraction_Violin.pdf",format='pdf')
    plt.close('all')


#Calculate statistics
for idx2 in range(len(labels)):
    d1  = d_list[idx2]
    f1  = f_list[idx2]
    n1  = len(d1)
    d1avg = np.average(d1)
    d1std = np.std(d1)
    d1sem = d1std/np.sqrt(n1)

    f1avg = np.average(f1)
    f1std = np.std(f1)
    f1sem = f1std/np.sqrt(n1)

    for idx3 in range(idx2+1,len(labels)):
        d2  = d_list[idx3]
        f2  = f_list[idx3]
        n2  = len(d2)
        d2avg = np.average(d2)
        d2std = np.std(d2)
        d2sem = d2std/np.sqrt(n2)

        f2avg = np.average(f2)
        f2std = np.std(f2)
        f2sem = f2std/np.sqrt(n2)

        pstatd= ttest(d1avg,d1sem,n1,d2avg,d2sem,n2)
        pstatf= ttest(f1avg,f1sem,n1,f2avg,f2sem,n2)
        print("Comparing "+labels[idx2]+" with "+labels[idx3]+":")
        print("\t"+labels[idx2]+", Deff: "+str(np.around(d1avg,decimals=3))+" +/- "+str(np.around(d1sem,decimals=3)))
        print("\t"+labels[idx3]+", Deff: "+str(np.around(d2avg,decimals=3))+" +/- "+str(np.around(d2sem,decimals=3)))
        print("\tp-value: "+str(np.around(pstatd,decimals=5))+"\n")
        print("\t"+labels[idx2]+", Mob. Frac.: "+str(np.around(f1avg,decimals=3))+" +/- "+str(np.around(f1sem,decimals=3)))
        print("\t"+labels[idx3]+", Mob. Frac.: "+str(np.around(f2avg,decimals=3))+" +/- "+str(np.around(f2sem,decimals=3)))
        print("\tp-value: "+str(np.around(pstatf,decimals=5))+"\n")
