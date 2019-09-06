import torch 
import torch.nn as nn
from IPython import embed
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers , num_classes):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.FC = nn.Sequential(
	        nn.Linear(hidden_size, 256),
		    nn.ReLU(),
		    nn.Linear(256, 128),
		    nn.ReLU(),		
            nn.Linear(128, num_classes),
		)
        self.num_layers = num_layers
		
    def forward(self, x, h, c):
        output, (hn, cn) = self.lstm(x, (h, c))
        h = output[:,-1,:]
        output = self.FC(h)
        return output

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))