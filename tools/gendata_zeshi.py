from ntu_read_skeleton import *
import os

def gendata(file_path, save_path, ignored_sample_path):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() +".skeleton" for line in f.readlines()
            ]
    file_list = os.listdir(file_path)
    #print(file_list)
    
    save_list=[]
    for i in range(len(file_list)):
        save_name = save_path + file_list[i].split('.')[0] + ".npy"
        if(file_list[i] in ignored_samples):
            continue
        file_name = file_path + file_list[i]
        data_xyz, data_root , class_index= read_xyz(file_name, rotation = True)
        #data = convertdata4render(data_xyz)
        data = convertdata4me(data_xyz, class_index)
        data = DIF(data,data_root, file_name)
        np.save(save_name, data)
    print("finished")






if __name__ == "__main__":
    gendata("../../nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/", "../../data_zeshi_DIF_p+r_2_select/", "./missing_skeletons.txt")