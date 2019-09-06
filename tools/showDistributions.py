import matplotlib.pyplot as plt
import numpy as np

'''
show differnet classes' distrbutions
'''
def showDistribution(class_list, file_path):
    data=[]
    for i in range(len(class_list)):
        file_name = file_path+'/'+str(class_list[i])+'.txt'
        data_class = np.loadtxt(file_name)
        data.append(data_class)
    plt.figure()
    for i in range(len(data)):
        plt.hist(data[i], bins =100, normed = 1,  label = str(class_list[i]), alpha = 0.5)
    plt.legend(loc = 'upper right')
    plt.show()


if __name__ == "__main__":
    file_path = './shape_motions'
    class_list = [11,12,30,29]
    showDistribution(class_list, file_path)