import numpy as np
import os
from IPython import embed
import glm
import math
import pybullet as p
def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            temp = f.readline()
            #print(temp)
            frame_info['numBody'] = int(temp)
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence
def motionNormalize(data):
    '''
    input data shape: (C, T, V, M)
    output data shape: (C, 300, V, M)
    '''
    #compute the number of frames between 
    T = data.shape[1]
    num_interplate_frame = int((300-data.shape[1])/(data.shape[1]-1))+1
    num_all_frames = data.shape[1]+(data.shape[1]-1)*num_interplate_frame
    data_temp = np.zeros((7, num_all_frames, 25, 2))
    # 2 actors
    for i in range(2):
        for j in range(T):
            if(j == (T-1)):
                data_temp[:, -1, :, :] = data[:, -1, :, :].copy()
            else:
                data_frame_pre = data[:, j, :, i]
                data_frame_aft = data[:, j+1, :, i]
                for k in range(num_interplate_frame+1):
                    alpha = 1.0/(num_interplate_frame+1) * k
                    for joint_id in range(25):
                        joint_pos_pre = data_frame_pre[0:3, j, joint_id, i]
                        joint_rotation_pre = data_frame_pre[3:7, j, joint_id, i]
                        joint_rotation_pre = [joint_rotation_pre[1], joint_rotation_pre[2], joint_rotation_pre[3], joint_rotation_pre[0]]

                        joint_pos_aft = data_frame_aft[0:3, j, joint_id, i]
                        joint_rotation_aft = data_frame_aft[3:7, j, joint_id, i]
                        joint_rotation_aft = [joint_rotation_aft[1], joint_rotation_aft[2], joint_rotation_aft[3], joint_rotation_aft[0]]








    return data_output
def read_skeleton_num(file):
    num_actors = []
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            temp = f.readline()
            #print(temp)
            frame_info['numBody'] = int(temp)
            num_actors.append(frame_info['numBody'])
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return np.max(np.array(num_actors))

def read_xyz(file, max_body=3, num_joint=25, rotation = False):
    '''
    return data shape:7, num_frame, num_joints, max_body
    '''

    print(file)
    seq_info = read_skeleton(file)
    class_index = int(file[-12:-9])
    print("class:{}".format(class_index))
    if(rotation == True):
        data_xyz = np.zeros((7, seq_info['numFrame'], num_joint, max_body))
    else:
        data_xyz = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    data_root = np.zeros((seq_info["numFrame"],4, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    if(rotation == True):
                        data_xyz[:, n, j, m] = [v['x'], v['y'], v['z'], 0, 0, 0, 0]
                        quat = np.array([v['orientationW'], v['orientationX'], v['orientationY'], v['orientationZ']])
                        if(np.linalg.norm(quat)<0.001):
                            quat = np.array([1, 0, 0, 0])
                        data_xyz[3:7, n, j, m] = quat
                    else:
                        data_xyz[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
            if (m < max_body):
                data_root[n,0, m] = f['bodyInfo'][m]['jointInfo'][0]['orientationW']
                data_root[n,1, m] = f['bodyInfo'][m]['jointInfo'][0]['orientationX']
                data_root[n,2, m] = f['bodyInfo'][m]['jointInfo'][0]['orientationY']
                data_root[n,3, m] = f['bodyInfo'][m]['jointInfo'][0]['orientationZ']
    return data_xyz, data_root, class_index
def convertdata4render(data):
    '''
    convert the data into (num_frame, 3*num_joints) shape, which is easy for rendering and debugging
    '''
    data = data[:, :, :, 0]
    data = np.transpose(data, (1,2,0))
    data = data.reshape((data.shape[0], -1))

    return data
def convertdata4me(data, class_index):
    '''
    convert the data into basic form and choose 2 possible actors
    input shape: (C, T, V, M)
    '''
    if(class_index<50):
        actors = 1
    else:
        actors = 2
    data_temp = data.copy()[0:3, :, :, :]
    data_std = np.sum( np.sum(np.std(data_temp, axis = 1), axis =0), axis =0)
    #data_std shape: (3,)
    dtype = [('id', int), ('std', float)]
    std_list = np.array([(0, data_std[0]) , (1, data_std[1]), (2, data_std[2])], dtype = dtype)
    std_list = np.sort(std_list, order = 'std')
    if(actors == 1):
        id_list = [std_list[-1][0]]
    if(actors == 2):
        id_list = [std_list[-1][0], std_list[-2][0]]
    data_temp = np.zeros((data.shape[0], data.shape[1], data.shape[2], 2))
    for i in range(actors):
        data_temp[:, :, :,i] = data[:, :, :, id_list[i]].copy()
    data_output = np.zeros((data.shape[1], data.shape[0] * data.shape[2], 2))
    for i in range(2):
        data_actor = data_temp[:, :, :, i].copy()
        data_actor = np.transpose(data_actor, (1,2,0))
        data_actor = data_actor.reshape((data_actor.shape[0], -1))
        data_output[:, :, i] = data_actor.copy()
    return data_output


def DIF(data, data_root, filename):
    '''
    convert global feature to DIF feature
    '''
    id_root = 0
    id_leftfoot = 19
    id_right_foot = 16
    if(np.sum(np.abs(data[:,:,1]))<0.001):
        num_actor = 1
    else:
        num_actor = 2
    print("num_actor:{}".format(num_actor))
    for actor in range(num_actor):
        for i in range(data.shape[0]):
            quat_root  = glm.quat(data_root[i,0,actor], data_root[i,1,actor], data_root[i,2,actor], data_root[i,3,actor])
            R_root = glm.mat4_cast(quat_root)
            x_axis = R_root*glm.vec4(1,0,0,1)
            x_axis_new = np.array([x_axis[0], 0, x_axis[2]])
            x_axis_new /= np.linalg.norm(x_axis_new)
            y_axis_new = np.array([0, 1, 0])
            z_axis_new = np.cross(x_axis_new, y_axis_new)


            R = np.eye(4)
            R[0:3,0] = x_axis_new
            R[0:3,1] = y_axis_new
            R[0:3,2] = z_axis_new
            T = np.eye(4)
            T[0,3] = data[i, 0, actor]
            T[2,3] = data[i, 2, actor]
            T[1,3] = 0.5*(data[i, id_leftfoot*3+1, actor]+data[i, id_right_foot*3+1, actor])

            M = np.linalg.inv(np.dot(T,R))

            rotation_matrix = glm.mat3([x_axis_new.tolist(), y_axis_new.tolist(), z_axis_new.tolist()])
            rotation_matrix = glm.mat4(rotation_matrix)

            
            for j in range(25):
                #embed()
                pos = np.array([data[i,j*7,actor], data[i, 7*j+1,actor], data[i, 7*j+2,actor],1])
                pos = np.dot(M, pos)

                quat = glm.quat(data[i, 7*j+3,actor], data[i, 7*j+4,actor], data[i, 7*j+5,actor], data[i, 7*j+6,actor])
                rotation_camera = glm.mat4_cast(quat)
                rotation_local = glm.inverse(rotation_matrix)*rotation_camera*rotation_matrix
                quat_local = glm.quat_cast(rotation_local)
                # make sure the W part of quaternion is larger than 0
                if(quat_local[3]<0):
                    quat_local = -quat_local 
                #print(np.linalg.norm(data[i,7*j+3:7*j+7]))
                for k in range(3):
                    data[i, 7*j+k, actor] = pos[k]
                for k in range(4):
                    data[i, 7*j+3+k, actor] = quat_local[(3+k)%4]
           
        #print("###")
    return data




if __name__=="__main__":
    data_xyz, data_root ,class_index= read_xyz("../../nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/S001C001P004R001A010.skeleton")
    data_xyz = data_xyz[0:3,:,:,:]
    data_xyz = convertdata4me(data_xyz, class_index)
    #data_xyz = DIF(data_xyz, data_root[:,:,0], "sss")
    np.save("E:/project2019/render/NTU_data/test.npy", data_xyz)