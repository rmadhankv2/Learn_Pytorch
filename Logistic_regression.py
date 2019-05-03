# import packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader as dloader

'''
STEP 1: LOAD DATASET
'''
trainDataset = dset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
testDataset = dset.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)


'''
STEP 2: MAKE DATASET ITERABLE
'''
batch_size = 100
n_iter = 3000
num_epochs = n_iter / (len(trainDataset) / batch_size)
num_epochs = 10
trainloader = dloader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
testloader = dloader(dataset=testDataset, batch_size=batch_size, shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 28*28
output_dim = 10

model = LogisticRegressionModel(input_dim, output_dim)

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
for epoch in range(num_epochs):
    for i, (image, labels) in enumerate(trainloader):
        # load image and variable
        if torch.cuda.is_available():
            images = Variable(image.view(-1, 28*28).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(image.view(-1, 28 * 28))
            labels = Variable(labels)

        # clear gradient for parameters
        optimizer.zero_grad()

        # forward pass to get outputs
        output = model(images)

        # calculate loss softmax --> cross entrophy
        loss = criterion(output, labels)

        # getting grad for parameters
        loss.backward()

        # updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # calculate accuracy
            correct = 0
            total = 0

            # iterate through test dataset
            for img, label in testloader:
                # load image to torch variable
                if torch.cuda.is_available():
                    imgs = Variable(img.view(-1, 28*28).cuda())
                    labels = Variable(label.cuda())
                else:
                    imgs = Variable(img.view(-1, 28 * 28))
                    labels = Variable(label)

                # forward pass to get output
                out = model(imgs)

                # get predictions from max value
                _, predicted = torch.max(out.data, 1)

                # total number of labels
                total += labels.size(0)

                # total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # print loss
            print('Iteration: {} || Loss: {} || Accuracy: {}'.format(iter, loss, accuracy))

# save model
save_model = False
if save_model is True:
    # saves only parameters
    # aplha & beta
    torch.save(model.state_dict(), 'LogisticRegressionModelState.pk1')

# load model
load_model = False
if load_model is True:
    model.load_state_dict(torch.load('LogisticRegressionModelState.pk1'))
