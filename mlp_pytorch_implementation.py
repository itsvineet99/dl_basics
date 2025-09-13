import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# loading iris dataset into memory
iris = load_iris()
X = iris['data']
y = iris['target']

# splitting data into train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=1./3, random_state=42)

# standardization of data
X_train_norm = (X_train - np.mean(X_train))/ np.std(X_train)

# transforming our data from numpy arrays to tensors
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)

# creating dataset by combining both features and target 
dataset = TensorDataset(X_train_norm, y_train)

# setting up batch size and creating dataloader object 
torch.manual_seed(42)
batch_size = 2

train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# creating MLP model to make predictions 
class Mlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x

# initializing model
input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3

model = Mlp(input_size, hidden_size, output_size)

# let's set up loss function and optimizer
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop for loss
num_epochs = 100
loss_hist = [0] * num_epochs
acc_hist = [0] * num_epochs

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[epoch] += loss.item()*y_batch.size(0) # by default loss is average over batch we convert it into sum of loss over batch
        is_correct = (torch.argmax(preds, dim=1) == y_batch).float()
        acc_hist[epoch] += is_correct.sum()
    print(f'Epoch {epoch} done.\n')
    loss_hist[epoch] /= len(train_dl.dataset) # averaging over entire dataset
    acc_hist[epoch] /= len(train_dl.dataset)


# plotting loss and accuracy 
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(acc_hist, lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

# saving trained model

path = 'iris_classifier.pt'
torch.save(model.state_dict(), path)