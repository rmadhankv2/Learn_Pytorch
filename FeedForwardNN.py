import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''
STEP 1: LOAD DATASET
'''
train_dataset = dset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dset.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

'''
STEP 2: MAKE DATASET ITERABLE
'''
batch_size = 100
epochs = 5
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        # linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # activation function
        self.relu = nn.ReLU()
        # linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # linear function
        out = self.fc1(x)
        # activation function
        out = self.relu(out)
        # linear function
        out = self.fc2(out)
        return out
'''
STEP 4: INSTANTIATE MODEL
'''
input_dim = 784
hidden_dim = 100
output_dim = 10

model = FeedForwardNN(input_dim, hidden_dim, output_dim)

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN MODEL
'''
iter = 0
loss_data = []
accuracy_data = []
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # load image and label in variable
        image = Variable(images.view(-1, 28*28))
        label = Variable(labels)

        # clear gradient for parameters
        optimizer.zero_grad()

        # forward pass to get output
        outputs = model(image)

        # calculate loss
        loss = criterion(outputs, label)

        # getting gradient parameters
        loss.backward()

        # updating parameters
        optimizer.step()

        iter += 1

        if iter % 100 == 0:
            # calculate accuracy
            correct = 0
            total = 0
            for imgs, lbls in test_loader:
                # load image
                img = Variable(imgs.view(-1, 28*28))

                # forward pass to get output
                outputs = model(img)

                # get predictions from mx value
                _, predicted = torch.max(outputs.data, 1)

                # total number of labels
                total += lbls.size(0)

                # total correct predictions
                correct += (predicted == lbls).sum()

            accuracy = 100 * correct / total
            loss_data.append(loss)
            accuracy_data.append(accuracy)
            # print loss
            print('Iteration : {}, Loss : {}, Accuracy : {}'.format(iter, loss.data, accuracy))

'''
PLOT LOSS AND ACCURACY
'''
plt.plot(range(30), loss_data)
plt.show()
