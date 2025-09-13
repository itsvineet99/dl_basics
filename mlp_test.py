import torch
import numpy as np
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# loading iris dataset into memory
iris = load_iris()
X = iris['data']
y = iris['target']

# splitting data into train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=1./3, random_state=42)

# data scaling 
X_test_norm = (X_test - np.mean(X_test))/np.std(X_test)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)

# mlp class implementation
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
input_size = X_test_norm.shape[1]
hidden_size = 16
output_size = 3

path = 'iris_classifier.pt'
new_model = Mlp(input_size, hidden_size, output_size)

# loading model parameters that we have saved on path
new_model.load_state_dict(torch.load(path))

# printing parameters of model 
print(new_model.eval())

# calculating predictions  and measuring test accuracy
pred_test = new_model(X_test_norm)

correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()

print(f'Test Accuracy: {accuracy:.4f}')