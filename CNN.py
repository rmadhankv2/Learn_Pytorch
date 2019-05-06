import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.autograd import Variable

'''
STEP 1: LOAD DATASET
'''
train_dataset = dset.MNIST('./data', download=True, train=True, transform=transforms.ToTensor())
test_dataset = dset.MNIST('./data', download=True, train=False, transform=transforms.ToTensor())

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
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Maxpooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Maxpooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Linear(32*7*7, 10)

    def forward(self, x):
        # Convo 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # maxpool 1
        out = self.maxpool1(out)

        # Convo 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # maxpool 1
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


'''
STEP 4: INSTANTIATE MODEL
'''
model = CNNModel()

if torch.cuda.is_available():
    model.cuda()

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
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        if torch.cuda.is_available():
            image = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            image = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()

        out = model(image)

        loss = criterion(out, labels)

        loss.backward()

        optimizer.step()

        iter += 1

        if iter % 100 == 0:
            # calculate accuracy
            correct = 0
            total = 0
            for imgs, lbls in test_loader:
                # load image
                img = Variable(imgs)

                # forward pass to get output
                outputs = model(img)

                # get predictions from mx value
                _, predicted = torch.max(outputs.data, 1)

                # total number of labels
                total += lbls.size(0)

                # total correct predictions
                correct += (predicted == lbls).sum()

            accuracy = 100 * correct / total
            # print loss
            print('Iteration : {}, Loss : {}, Accuracy : {}'.format(iter, loss.data, accuracy))

