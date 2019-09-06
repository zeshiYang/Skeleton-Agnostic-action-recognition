import numpy as np
import matplotlib.pyplot as plt

Tmean = np.loadtxt("./t_mean.txt")
acc = np.loadtxt("./acc.txt")
Tstd = np.loadtxt("./t_std.txt")


def takeSecond(elem):
    return elem[1]

Tmean = Tmean.tolist()
for i in range(len(Tmean)):
    Tmean[i] = (i+1, Tmean[i])
Tmean.sort(key = takeSecond)
id_list = []
for i in range(len(Tmean)):
    id_list.append(Tmean[i][0])
acc_sorted =[]
for i in range(len(id_list)):
    acc_sorted.append(acc[id_list[i]-1])
tstd_sorted =[]
for i in range(len(id_list)):
    tstd_sorted.append(Tstd[id_list[i]-1])
tavg_sorted =[]
for i in range(len(id_list)):
    tavg_sorted.append(Tmean[i][1])
x = np.arange(1,61)
f = []
    
ax = plt.subplot(3, 1, 1)       
pic = plt.plot(x, np.array(acc_sorted), linestyle = '--', marker = '*', linewidth = 2.0, label = 'accuracy')
plt.legend(loc = 'upper right', fontsize = 'large')
f.append(pic)

ax = plt.subplot(3, 1, 2)       
pic = plt.plot(x, np.array(tavg_sorted), linestyle = '--', marker = '*', linewidth = 2.0,label= 'Tmean')
plt.legend(loc = 'upper right', fontsize = 'large')
f.append(pic)

ax = plt.subplot(3, 1, 3)       
pic = plt.plot(x, np.array(tstd_sorted), linestyle = '--', marker = '*', linewidth = 2.0,label = 'Tstd')
plt.legend(loc = 'upper right', fontsize = 'large')
f.append(pic)
plt.show()
