import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from IPython import embed
import math
        


class MotionDataset(Dataset):
    def __init__(self, datadir, classes_train, classes_index, actors, views, channels = 7, num_frames = 300, useGCNN=True, preNormal = False, zeroPadding = False, mode = 'r'):
        '''
        classes: action classes
        actors: performers of actions
        views: camera views


        '''

        self.useGCNN = useGCNN
        self.Tmax = 0
        self.Tmin = 1000
        self.channels = channels
        self.preNormal = preNormal
        self.zeroPadding = zeroPadding
        
        def isTrain(file):
            '''
            decide wether a file is trainable
            '''
            flag_class = False
            flag_actor = False
            flag_view = False
            actor_index = int(file[9:12])
            class_index = int(file[17:20])
            view_index = int(file[5:8])
            for i in range(len(classes_train)):
                if(class_index == classes_train[i]):
                    flag_class = True
            for i in range(len(actors)):
                if(actor_index == actors[i]):
                    flag_actor = True
            for i in range(len(views)):
                if(view_index == views[i]):
                    flag_view = True

            return flag_actor and flag_class and flag_view
        
        def index(class_index, classes_index):
            for i in range(len(classes_index)):
                if(class_index == classes_index[i]):
                    return i


        self.num_frames = num_frames
        files_list = os.listdir(datadir)
        self.classes_train = classes_train
        self.actors = actors
        files_selected = list(filter(isTrain, files_list))
        self.num_data = len(files_selected)
        self.data = []
        #normalization
        data_raw = []
        #embed()
        T_list=[]
        for i in range(len(files_selected)):
            motion = np.load(datadir+"/"+files_selected[i])
            #if(math.isnan(motion.sum())):
            T = motion.shape[0]
            T_list.append(T)
            if(T>self.Tmax):
                self.Tmax = T
            if(T<self.Tmin):
                self.Tmin = T
            motion_padding = np.zeros((300, 25*(self.channels), 2))
            motion_padding[0:T, :, :] = motion.copy()
            if(self.zeroPadding == False):
                for j in range(T, 300):
                    motion_padding[j, :, :] = motion[-1,:,:].copy()
            if(useGCNN == True):
                motion = motion_padding
                motion = torch.Tensor(motion)
                motion = motion.view(motion.shape[0], -1, self.channels, motion.shape[2])
                motion = motion.permute(2,0,1,3)
                motion = motion.view(motion.shape[0], motion.shape[1], motion.shape[2], -1).numpy()
                if(mode == 'r'):
                    motion = motion[3:7, :, :, :]
                elif(mode == 'p'):
                    motion = motion[0:3, :, :, :]
                else:
                    pass
            #print(files_selected[i])
            class_index = int(files_selected[i][17:20])
            #print(class_index)
            class_index = index(class_index, classes_index)
            
            sample = {"motion": motion, "class": class_index}
            

           
            self.data.append(sample)
            data_raw.append(motion)
        if(not self.useGCNN):
            data_raw = np.concatenate(data_raw, 0)
            self.mean  = np.mean(data_raw, 0).reshape((self.channels, 1, 25, 1))
            self.std = np.std(data_raw, 0).reshape((self.channels, 1, 25, 1))
        else:
            #embed()
            #data_raw = np.concatenate(data_raw, 1)
            #self.mean  = np.mean(data_raw, 1).reshape((self.channels, 1, 25, 1))
            #self.std = np.std(data_raw, 1).reshape((self.channels, 1, 25, 1))
            pass
        self.T_avg=np.mean(np.array(T_list))
        self.T_std=np.std(np.array(T_list))
        print("get all the data")
        print("Tmax:{}".format(self.Tmax))
        print("Tmin:{}".format(self.Tmin))
        print("T_avg:{}".format(self.T_avg))
        print("T_std:{}".format(self.T_std))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        motion = sample["motion"]
        class_index = sample["class"]
        
        if(self.useGCNN):
            #start_index = np.random.randint(0, motion.shape[1]-self.num_frames)
            if(self.preNormal):
                motion_sample = (motion-self.mean)/(self.std+0.001)
            else:
                motion_sample = motion
        else:
            #start_index = np.random.randint(0, motion.shape[0]-self.num_frames)
            motion_sample = motion
        return motion_sample, class_index

    def setMeanStd(self, mean, std):
        self.mean = mean
        self.std = std


if __name__ =="__main__":
    train_dataset = MotionDataset("../data_zeshi", [1, 5, 6, 7, 8, 9, 10, 14, 16, 23], [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38],[2,3])
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    
    test_dataset = MotionDataset("../data_zeshi", [1, 5, 6, 7, 8, 9, 10, 14, 16, 23], [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40],[1])
    test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = True)
    for i, data in enumerate(train_dataloader, 0):
        #embed()
        motions, classes = data
        print(classes)
        


