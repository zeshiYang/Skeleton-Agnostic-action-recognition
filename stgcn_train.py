from feeder import *
from net.stgcn import * 
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
device = torch.device("cuda:1")
from IPython import embed

def test_model(test_dataloader, net, criteron, device):
    correct = 0
    total = 0
    loss = 0
    net.eval()
    with torch.no_grad():
        for data in test_dataloader:
            motions, classes = data
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
    print("Accuracy:{}".format(correct/total))
    print("CrossEntropy loss：{}".format(loss))
    return correct/total, loss
def adjust_learning_rate(optimizer, lr_init,  epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (0.1 ** (epoch // 10))
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









if __name__ =="__main__":
    batch_size = 32

    train_dataset = Feeder("./train_data.npy", "./train_label.pkl")
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 6)
    
    test_dataset = Feeder("./val_data.npy", "./val_label.pkl")
   
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 6)

    #net = Net(77, 256, 1, 10)
    graph_args = {"layout":"ntu-rgb+d", "strategy":"spatial"}
    kwargs = {"dropout":0.5}
    net = Model(3, 60, graph_args, False, **kwargs)
    net.apply(weights_init)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    lr_init = 0.01
    optimizer = optim.SGD(net.parameters(), lr_init)
    
   

    print("start training")
    writer = SummaryWriter()

    for epoch in range(80):
        running_loss = 0
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            motions, classes = data
            motions = motions.to(device).float()
            classes = classes.to(device).long()
            optimizer.zero_grad()
            outputs = net(motions)
            loss = criterion(outputs, classes)        
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("iter:{}".format(epoch))
        print("training loss:{}".format(running_loss))
        writer.add_scalar("data/training_loss", running_loss, epoch)
        adjust_learning_rate(optimizer, lr_init, i)

        if(epoch%10==0):
            print("training")
            acc, _ = test_model(train_dataloader, net, criterion, device)
            writer.add_scalar("data/training_accuracy", acc, epoch)
            print("testing")
            acc, loss =test_model(test_dataloader, net, criterion, device)
            writer.add_scalar("data/testing_loss", loss, epoch)
            writer.add_scalar("data/testing_accuracy", acc, epoch)


    
