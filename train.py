from motiondataset import *
from net.stgcn import * 
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch
from collections import OrderedDict

device = torch.device("cuda:0")
from IPython import embed

def test_model(test_dataloader, net, criteron, device):
    correct = 0
    total = 0
    loss = 0
    num_sample = 0
    res =[]
    net.eval()
    with torch.no_grad():
        for data in test_dataloader:
            motions, classes = data
            num_sample += motions.shape[0]
            motions = motions.to(device).float()
            classes = classes.to(device).long()
            
            
            #h0 = torch.Tensor(np.zeros((1, motions.size(0), 256))).to(device)
            #c0 = torch.Tensor(np.zeros((1, motions.size(0), 256))).to(device)
            #outputs = net(motions, h0, c0)
            outputs = net(motions)
            loss += criteron(outputs, classes)
            _, predicted = torch.max(outputs.data, 1)
            total += classes.size(0)
            correct += (predicted == classes).sum().item()
            res += predicted.cpu().numpy().tolist()
    
    print("Accuracy:{}".format(correct/total))
    print("CrossEntropy loss：{}".format(loss/num_sample))
    return correct/total, loss, np.array(res)
def adjust_learning_rate(optimizer, lr_init,  epoch, steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (0.1 ** (np.sum(epoch>steps)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_weights(model, weights_path, ignore_weights=None):
    if ignore_weights is None:
        ignore_weights = []
    if isinstance(ignore_weights, str):
        ignore_weights = [ignore_weights]

    print('Load weights from {}.'.format(weights_path))
    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1],
                            v.cpu()] for k, v in weights.items()])

    # filter weights
    for i in ignore_weights:
        ignore_name = list()
        for w in weights:
            if w.find(i) == 0:
                ignore_name.append(w)
        for n in ignore_name:
            weights.pop(n)
            print('Filter [{}] remove weights [{}].'.format(i,n))

    for w in weights:
        print('Load weights [{}].'.format(w))

    try:
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        for d in diff:
            print('Can not find weights [{}].'.format(d))
        state.update(weights)
        model.load_state_dict(state)
    return model







if __name__ =="__main__":
    batch_size = 32
    actors=[]
    for i in range(40):
        actors.append(i+1)
    #actions = [1, 5, 6, 7, 8, 9, 10, 14, 16, 23, 2, 3, 4, 11, 12, 13, 15, 17, 20, 21, 18, 19, 22, 24, 25, 26, 27, 28, 29, 30]
    #actions = [1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23]
    actions = [1,2,3,5,6,7,8,9,10,19,20,21,22,23,24,25,26,27,28,31]
    #train_dataset = MotionDataset("../data_zeshi_DIF_camera", [1, 5, 6, 7, 8, 9, 10, 14, 16, 23], [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38])
    train_dataset = MotionDataset("../data_zeshi_DIF_p+r",actions ,actions, [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38], [1,2,3], channels = 7, zeroPadding = True)
    #train_dataset = MotionDataset("../data_zeshi_DIF_camera", [1, 5, 6, 7, 8, 9, 10, 14, 16, 23], actors, [2,3])
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 12)
    
    #test_dataset = MotionDataset("../data_zeshi_DIF_camera", [1, 5, 6, 7, 8, 9, 10, 14, 16, 23], [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40])
    test_dataset = MotionDataset("../data_zeshi_DIF_p+r", actions, actions, [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40], [1,2,3], channels = 7, zeroPadding = True )
    #test_dataset = MotionDataset("../data_zeshi_DIF_camera", [1, 5, 6, 7, 8, 9, 10, 14, 16, 23], actors,[1] )
    #test_dataset.setMeanStd(train_dataset.mean, train_dataset.std)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 12)

    #net = Net(77, 256, 1, 10)
    graph_args = {"layout":"ntu-rgb+d", "strategy":"spatial"}
    kwargs = {"dropout":0.5}
    net = nn.DataParallel(Model(4, 20, graph_args, True, **kwargs))
    net.apply(weights_init)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum =0.9, nesterov = True, weight_decay=0.0001)
    #optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=0.0001)
    steps = np.array([10, 50])
   

    print("start training")
    writer = SummaryWriter("./feature_test/padding_test/cross_subject_DIF_noschedule_r_300_zero_padding_imp")

    for epoch in range(100):
        running_loss = 0
        net.train()
        num_sample = 0
        for i, data in enumerate(train_dataloader, 0):
            motions, classes = data
            #embed()
            num_sample += motions.shape[0]
            motions = motions.to(device).float()
            classes = classes.to(device).long()
            optimizer.zero_grad()
            #embed()
            #h0 = torch.Tensor(np.zeros((1, motions.size(0), 256))).to(device)
            #c0 = torch.Tensor(np.zeros((1, motions.size(0), 256))).to(device)
            #outputs = net(motions, h0, c0)
            outputs = net(motions)
            
            loss = criterion(outputs, classes)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("iter:{}".format(epoch))
        print("mean training loss:{}".format(running_loss/num_sample))
        writer.add_scalar("data/training_loss", running_loss, epoch)
        adjust_learning_rate(optimizer, 0.1, epoch, steps)

        if(epoch%10==0):
            print("testing")
            acc, loss ,_=test_model(test_dataloader, net, criterion, device)
            writer.add_scalar("data/testing_loss", loss, epoch)
            writer.add_scalar("data/testing_accuracy", acc, epoch)

            print("training")
            acc, _, _= test_model(train_dataloader, net, criterion, device)
            writer.add_scalar("data/training_accuracy", acc, epoch)
        

    
