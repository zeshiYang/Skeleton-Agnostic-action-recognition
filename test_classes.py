from train import *
from feeder import *
import matplotlib.pyplot as plt
import numpy as np

'''
in this file, I would test classification accuracy of 60 different classes to see
wether durations of motion will affect classification accuracy
'''







if __name__ == "__main__":
    batch_size = 32
   
    #test_dataset = Feeder("./data_st-gcn/xsub/val_data.npy", "./data_st-gcn/xsub/val_label.pkl")
   
    #test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 6)
    actions =[]
    for i in range(60):
        actions.append(i+1)
    actors = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]


    weights_path = '../model/st_gcn.ntu-xsub-300b57d4.pth'


    #net = Net(77, 256, 1, 10)
    graph_args = {"layout":"ntu-rgb+d", "strategy":"spatial"}
    kwargs = {"dropout":0.4}
    net = Model(3, 60, graph_args, True, **kwargs)
    weight_path = "model path"
    load_weights(net, weights_path)
    print("model loaded")
    net.to(device)
    criterion = nn.CrossEntropyLoss()   
    acc_list=[]
    num_list = []
    T_mean=[]
    T_std=[]
    res_list=[]
    for i in range(60):
        print("class :{}".format(i+1))
        
        test_dataset = MotionDataset("../data_zeshi_st-gcn", [i+1], actions, actors, [1,2,3], channels = 3, zeroPadding = True )
        print("num_trainingdata:{}".format(test_dataset.num_data))
        num_list.append(test_dataset.num_data)
        T_mean.append(test_dataset.T_avg)
        T_std.append(test_dataset.T_std)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 6)
        acc, loss, res =test_model(test_dataloader, net, criterion, device)
        acc_list.append(acc)
        res_list.append(res)
        #print("testing_loss:{}".format(loss))
        #print("testing_accuracy:{}".format(acc))
        print("#####")
    np.savetxt("./acc.txt", np.array(acc_list))
    np.savetxt("./t_mean.txt", np.array(T_mean))
    np.savetxt("./t_std.txt", np.array(T_std))
    if(False):
        f = []
       
        ax = plt.subplot(3, 1, 1)       
        pic = plt.plot(np.arange(len(acc_list)), np.array(acc_list), linestyle = '--', marker = '*', linewidth = 2.0, label = 'accuracy')
        f.append(pic)

        ax = plt.subplot(3, 1, 2)       
        pic = plt.plot(np.arange(len(acc_list)), np.array(T_mean), linestyle = '--', marker = '*', linewidth = 2.0,label= 'Tmean')
        f.append(pic)

        ax = plt.subplot(3, 1, 3)       
        pic = plt.plot(np.arange(len(acc_list)), np.array(T_std), linestyle = '--', marker = '*', linewidth = 2.0,label = 'Tstd')
        f.append(pic)
        plt.show()
    if(True):
       
        for i in range(60):
            fig = plt.figure()
            pic = plt.hist(res_list[i], bins =280, label = str(i+1))
            plt.legend(loc = 'upper right', fontsize = 'large')
            plt.savefig("./"+str(i)+".png")
            plt.close()

  

    