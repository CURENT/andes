import os
import csv
import glob
from numpy import *
import pandas as pd
import numpy as np
# os.system('python run.py')
# os.system('andes GT_* --ncpu=3')
os.chdir('C:/Users/zhan2/PycharmProjects/andes_github/demos/detect/GT')
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
    df=pd.DataFrame(voltage)
    df.to_csv('%s_voltage.csv' % filename,index=False)
    df=pd.DataFrame(frequency)
    df.to_csv('%s_frequency.csv' % filename,index=False)
    # output_file=open('%s.csv' % filename,'w')
    # with output_file:
    #     writer = csv.writer(output_file)
    #     writer.writerows(time)
    print('pp')
                # frequency(idx).append(ddc[bus_idx])
                # bus_idx.append(idx)
                # voltage(idx).append(ddc[])

# with open ('GT_100121_Syn2_5_out.dat') as f:
#     reader = csv.reader(f, delimiter="\t")
#     for line in reader:
#         print(line[7])