import os
import h5py
import numpy as np
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# Initialize variables to store the EEG data and labels
eeg_data_list = []
labels = []
# num_val_samples = 8
# num_test_samples = 10

# Replace 'folder_path' with the path to your data folder
folder_path = '/Users/siddharth/Downloads/EEG Data Associated with Mental Workload/'

# Loop through the first 30 files in the folder to collect EEG data and labels
# num_samples = 30
# samples_collected = 0

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*7, 1)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.reshape(-1, 4*2*7)
        x = F.sigmoid(self.fc1(x))
        return x


net = EEGNet()
print (net.forward(Variable(torch.Tensor(np.random.rand(1, 1, 120, 64)))))
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 100
    
    predicted = []
    
    for i in range(len(X)//batch_size):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)
        
        predicted.append(pred.datanumpy())
        
        
    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    
    predicted = predicted.datanumpy()
    
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))
    return results

  
num_train_samples = 30
num_val_samples = 8
num_test_samples = 10

# Loop through the files in the folder to collect EEG data and labels
samples_collected = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        mat_file = os.path.join(folder_path, filename)
        with h5py.File(mat_file, 'r') as mat_data:
            eeg_data = np.transpose(mat_data['EEG'][:])
            eeg_data_list.append(eeg_data)
            labels.append(samples_collected)  # Use an appropriate label for each sample
            samples_collected += 1

# Combine all the collected data and labels
eeg_data = np.stack(eeg_data_list, axis=0)  # Stack along the first dimension
labels = np.array(labels)

# Shuffle the data and labels randomly to ensure randomness in the splits
rng_state = np.random.get_state()
np.random.shuffle(eeg_data)
np.random.set_state(rng_state)
np.random.shuffle(labels)

# Split the data and labels into training, validation, and testing sets
X_train = eeg_data[:num_train_samples]
Y_train = labels[:num_train_samples]
Y_train = Y_train.reshape(-1, 1)
# X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1], X_train.shape[2])
X_val = eeg_data[num_train_samples:num_train_samples + num_val_samples]
# X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
Y_val = labels[num_train_samples:num_train_samples + num_val_samples]
Y_val = Y_val.reshape(-1, 1)

X_test = eeg_data[num_train_samples + num_val_samples:num_train_samples + num_val_samples + num_test_samples]
# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

Y_test = labels[num_train_samples + num_val_samples:num_train_samples + num_val_samples + num_test_samples]
Y_test = Y_test.reshape(-1, 1)

# Ensure the data type is float32 for X_train, X_val, and X_test
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# Print the shapes of the resulting datasets to verify
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


batch_size = 32

for epoch in range(10):  # loop over the dataset multiple times
    print ("\nEpoch ", epoch)
    
    running_loss = 0.0
    for i in range(len(X_train)//batch_size-1):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = torch.from_numpy(X_train[s:e])
        labels = torch.FloatTensor(np.array([y_train[s:e]]).T*1.0)
        
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        
        optimizer.step()
        
        running_loss += loss.data[0]
    
    params = ["acc", "auc", "fmeasure"]
    print(params)
    print( "Training Loss ", running_loss)
    print ("Train - ", evaluate(net, X_train, Y_train, params))
    print ("Validation - ", evaluate(net, X_val, Y_val, params))
    print ("Test - ", evaluate(net, X_test, Y_test, params))