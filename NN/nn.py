import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import pandas as pd
import random,time
from sklearn.preprocessing import StandardScaler
import sys


inputFileName = "input.csv"
dropLastN = 0
subsetFrac = 0.7
testFrac = 0.2
learning_rate = 0.001
weight_decay = 0.0
batch_size = 128
num_epochs = 30

args = sys.argv[1:]
if not(len(args) == 0 or len(args) == 3):
    print("usage is: nn.py or nn.py <input file> <drop last n columns> <data fraction>")

if len(args) == 3:
    inputFileName = args[0]
    dropLastN = int(args[1])
    subsetFrac = float(args[2])




filenum = int(round(time.time() * 1000))
open(f"{filenum}_code.py", "a").write(open(__file__).read())

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f"{filenum}_output.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()
        pass    
sys.stdout = Logger()


np.random.seed(25)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RmseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1) 
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = F.relu(out)

        out = self.fc4(out)
        out = F.relu(out)

        out = self.fc5(out)
        #out = torch.clamp(out, 1.0, 5.0)
        return out

df = pd.read_csv(inputFileName, sep=",")
df_percent = df.sample(frac=1).reset_index(drop=True).sample(frac=subsetFrac)

train=df_percent.sample(frac= 1.0 - testFrac)
test=df_percent.drop(train.index)


train_labels = torch.tensor(np.expand_dims(train['Stars'].values.astype(np.float32), axis=1))
train_temp = train.drop('Stars', axis=1) if dropLastN == 0 else train.drop('Stars', axis=1).iloc[:, :-dropLastN] #
train_norm = StandardScaler().fit_transform(train_temp)
train_features = torch.tensor(train_norm.astype(np.float32))
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_features, train_labels), batch_size = batch_size, shuffle = False, pin_memory=True)

test_labels = torch.tensor(np.expand_dims(test['Stars'].values.astype(np.float32), axis=1))
test_temp = test.drop('Stars', axis=1) if dropLastN == 0 else test.drop('Stars', axis=1).iloc[:, :-dropLastN] #
test_norm = StandardScaler().fit_transform(test_temp)
test_features = torch.tensor(test_norm.astype(np.float32))
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_features, test_labels), batch_size = batch_size, shuffle = False, pin_memory=True)




input_size = train_norm.shape[1]
model = NeuralNet(input_size).to(device)

criterion = RmseLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


epoch = 0
isContinue = True
while epoch < num_epochs and isContinue:
    start = time.time()
    total_loss = 0.0
    rounds = 0
    train_correct = 0
    train_total = 0
    for i, (images, labels) in enumerate(train_loader):  
        rounds += 1
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        # print(criterion(images, labels).data.cpu().numpy())
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs + 0.0000001, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data.cpu().numpy()
    test_rounds = 0
    test_total_loss = 0.0

    for i, (images, labels) in enumerate(test_loader):  
        test_rounds += 1
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        # print(criterion(images, labels).data.cpu().numpy())
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs + 0.0000001, labels)
        
        test_total_loss += loss.data.cpu().numpy()



    end = time.time()
    #Print your results every epoch
    print (f'Epoch: {str(epoch + 1).zfill(int(np.log10(num_epochs + 1) + 1))}, Average Batch Loss: {(total_loss / rounds):.4f}, Validation Avg. Batch Loss: {(test_total_loss / test_rounds):.4f}, It took: {(end-start):.1f}')
    epoch += 1