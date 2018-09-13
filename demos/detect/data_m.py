import os
import csv
import glob
from numpy import *
import pandas as pd
import numpy as np
os.system('andes -C')
os.system('python run.py')
# ==================== Generator trip ========================#
os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/NPCC/GT')
os.system('andes GT_*.dm -r tds --tf 10 --ncpu=8')
bus_count=995
for filename in glob.glob('*.dat'):
    with open (filename) as f:
        reader = csv.reader(f, delimiter="\t")
        raw_data=list(reader)
    row_count = sum(1 for row in raw_data)
    frequency=mat(zeros((row_count, bus_count)))
    voltage = mat(zeros((row_count, bus_count)))
    time=mat(zeros((row_count,1)))
    for idx,line in enumerate(raw_data):
            ddc=line[0].split()
            time[idx,0]=ddc[0]
            for idx_2,freq_idx in enumerate(range(996,1991)):
                frequency[idx,idx_2]=ddc[freq_idx]
            for idx_3, vol_idx in enumerate(range(3981, 4976)):
                voltage[idx, idx_3] = ddc[vol_idx]
    voltage=np.concatenate((time,voltage),axis=1)
    frequency=np.concatenate((time,frequency),axis=1)

    os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/Output/GT')
    df=pd.DataFrame(voltage)
    df.to_csv('%s_voltage.csv' % filename,index=False)
    df=pd.DataFrame(frequency)
    df.to_csv('%s_frequency.csv' % filename,index=False)
    os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/GT')
# ==================== Load shedding ========================
os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/LS')
os.system('andes LS_*.dm -r t --tf 0.1 --ncpu=8')
bus_count=995
for filename in glob.glob('*.dat'):
    with open (filename) as f:
        reader = csv.reader(f, delimiter="\t")
        raw_data=list(reader)
    row_count = sum(1 for row in raw_data)
    frequency=mat(zeros((row_count, bus_count)))
    voltage = mat(zeros((row_count, bus_count)))
    time=mat(zeros((row_count,1)))
    for idx,line in enumerate(raw_data):
            ddc=line[0].split()
            time[idx,0]=ddc[0]
            for idx_2,freq_idx in enumerate(range(996,1991)):
                frequency[idx,idx_2]=ddc[freq_idx]
            for idx_3, vol_idx in enumerate(range(3981, 4976)):
                voltage[idx, idx_3] = ddc[vol_idx]
    voltage=np.concatenate((time,voltage),axis=1)
    frequency=np.concatenate((time,frequency),axis=1)

    os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/Output/LS')
    df=pd.DataFrame(voltage)
    df.to_csv('%s_voltage.csv' % filename,index=False)
    df=pd.DataFrame(frequency)
    df.to_csv('%s_frequency.csv' % filename,index=False)
    os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/LS')

# ==================== Line trip ========================
os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/LT')
os.system('andes LT_*.dm -r t --tf 0.1 --ncpu=8')
bus_count=995
for filename in glob.glob('*.dat'):
    with open (filename) as f:
        reader = csv.reader(f, delimiter="\t")
        raw_data=list(reader)
    row_count = sum(1 for row in raw_data)
    frequency=mat(zeros((row_count, bus_count)))
    voltage = mat(zeros((row_count, bus_count)))
    time=mat(zeros((row_count,1)))
    for idx,line in enumerate(raw_data):
            ddc=line[0].split()
            time[idx,0]=ddc[0]
            for idx_2,freq_idx in enumerate(range(996,1991)):
                frequency[idx,idx_2]=ddc[freq_idx]
            for idx_3, vol_idx in enumerate(range(3981, 4976)):
                voltage[idx, idx_3] = ddc[vol_idx]
    voltage=np.concatenate((time,voltage),axis=1)
    frequency=np.concatenate((time,frequency),axis=1)

    os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/Output/LT')
    df=pd.DataFrame(voltage)
    df.to_csv('%s_voltage.csv' % filename,index=False)
    df=pd.DataFrame(frequency)
    df.to_csv('%s_frequency.csv' % filename,index=False)
    os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/LT')