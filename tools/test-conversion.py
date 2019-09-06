from ntu_read_skeleton import *
import torch
from IPython import embed

'''
test whether the data format conversion is correct or not
'''

if __name__ == "__main__":
    data_xyz, data_root = read_xyz("../../nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/S001C001P001R001A002.skeleton")


    data_zeshiformat = convertdata4me(data_xyz)
    motion = torch.Tensor(data_zeshiformat)
    motion = motion.view(motion.shape[0], -1, 3, motion.shape[2])
    motion = motion.permute(2,0,1,3)
    motion = motion.view(motion.shape[0], motion.shape[1], motion.shape[2], -1).numpy()

    print(np.sum(np.abs(motion - data_xyz)))
    #embed()