import os
import matplotlib.pyplot as plt
import numpy as np
import json
from IPython import embed
'''
in this script we show durations of different motions
'''

def showMotionTime(file_path, plot = False):
    file_list = os.listdir(file_path)
    num_classes = 60
    T_list =[]
    name_list = []
    for i in range(num_classes):
        T_list.append([])
        name_list.append([])
  
    for i in range(len(file_list)):
        file = file_list[i]
        class_index = int(file[17:20])
        motion = np.load(file_path + '/' + file)
        t = motion.shape[0]
        T_list[class_index-1].append(t)
        name_list[class_index-1].append(file)

    print("finish data collecting")
    motion_dict = {}
    for i in range(num_classes):
        for j in range(len(T_list[i])):
            motion_dict[name_list[i][j]] = T_list[i][j]
    with open('./data_distribution.json', 'w') as f:
        json.dump(motion_dict, f)
    for i in range(num_classes):
        np.savetxt("./shape_motions/"+str(i+1)+".txt", np.array(T_list[i]))



    

    if(plot):
        f = []
        for i in range(num_classes):
            ax = plt.subplot(6, 10, i+1)
            pic = ax.hist(np.array(T_list[i]), bins =100, label = str(i+1))
            plt.legend(loc = 'upper right', fontsize = 'large')
            f.append(pic)
        plt.show()

if __name__ == "__main__":
    showMotionTime("../../data_zeshi_DIF_camera")
