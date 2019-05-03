'''
STEP 1: GET DATA
'''

import numpy as np

x_values = [i for i in range(11)]

x_train = np.array(x_values, dtype=np.float32)

x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]

y_train = np.array(y_values, dtype=np.float32)

y_train = y_train.reshape(-1, 1)

'''
STEP 2: CREATE MODEL CLASS
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

# model blue print
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

'''
STEP 3: INSTANTIATE MODEL CLASS
'''

input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

# move to gpu if available
if torch.cuda.is_available():
    model.cuda()

'''
STEP 4: INSTANTIATE LOSS CLASS
'''

# loss function
criterion = nn.MSELoss()

'''
STEP 5: INSTANTIATE OPTIMIZER CLASS
'''

# instatiate optimizer class
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 6: TRAIN THE MODEL
'''

# training the model
epochs = 200

for epoch in range(epochs):
    epoch += 1 # for output console

    # convert numpy array to torch Variable (move the variables to gpu if available)
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # clear gradient for parameters
    optimizer.zero_grad()

    # forward to get output
    outputs = model.forward(inputs)

    # calculate loss
    loss = criterion(outputs, labels)

    # getting gradients for parameter
    loss.backward()

    # updating parameters
    optimizer.step()

    print('Epoch {}, Loss {}'.format(epoch, loss.data))

# get prediction
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

import matplotlib.pyplot as plt

# plot true data
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

# plot prediction
plt.plot(x_train, predicted, '--', label='Prediction', alpha=0.5)

plt.legend(loc='best')
plt.show()

# save model
save_model = False
if save_model is True:
    # saves only parameters
    # aplha & beta
    torch.save(model.state_dict(), 'LinearRegressionModelState.pk1')

# load model
load_model = False
if load_model is True:
    model.load_state_dict(torch.load('LinearRegressionModelState.pk1'))