from ntu_read_skeleton import *
import torch
from IPython import embed

'''
test whether the data format conversion is correct or not
'''

if __name__ == "__main__":
    data_xyz, data_root = read_xyz("../../nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/S001C001P001R001A002.skeleton", rotation = True)


    data_zeshiformat = convertdata4me(data_xyz)
    data_dif = DIF(data_zeshiformat.copy(), data_root, "001")


    data_xyz1, data_root1 = read_xyz("../../nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/S001C002P001R001A002.skeleton", rotation = True)


    data_zeshiformat1 = convertdata4me(data_xyz1)
    data_dif1 = DIF(data_zeshiformat1.copy(), data_root1, "002")
    embed()
