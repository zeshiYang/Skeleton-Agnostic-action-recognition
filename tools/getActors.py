from ntu_read_skeleton import *
import os
from IPython import embed
'''
get the max number of actors in kinect data
'''
def genMaxActors(file_path, ignored_sample_path):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() +".skeleton" for line in f.readlines()
            ]
    file_list = os.listdir(file_path)
    #print(file_list)
    num_actors =[]
    save_list=[]
    for i in range(len(file_list)):
        if(file_list[i] in ignored_samples):
            continue
        file_name = file_path + file_list[i]
        num_actors.append(read_skeleton_num(file_name))
    num_actors = np.array(num_actors)
    embed()
    print("max actors:{}".format(num_actors))
    print("finished")






if __name__ == "__main__":
    genMaxActors("../../nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/", "./missing_skeletons.txt")