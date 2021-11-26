from random import shuffle
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class Classifier():
    def __init__(self, batch_size, lr=0.001):
        self.net = LeNet()

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=lr)


        transform=transforms.Compose([
        # Pad images with 0s
        transforms.Pad((0,4,4,0), fill=0, padding_mode='constant'),
    
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset_full = datasets.MNIST('./mnist_data', train=True, download=True,
                        transform=transform)
        valid_size = 5000
        train_size = len(dataset_full) - 5000
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset_full, [train_size, valid_size])

        dataset_test = datasets.MNIST('./mnist_data', train=False,
                        transform=transform)

        self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(dataset_test, batch_size, shuffle=False)


    # Train & test part from https://github.com/activatedgeek/LeNet-5
    def train(self, num_epochs=30):
        for e in range(num_epochs):
            self.net.train()
            loss_list, batch_list = [], []
            for i, (images, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.net(images)
                loss = self.criterion(output, labels)

                loss_list.append(loss.detach().cpu().item())
                batch_list.append(i+1)

                #if i % 10 == 0:
                #    print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

                loss.backward()
                self.optimizer.step()

            validation_accuracy , validation_predictions = self.evaluate(self.valid_loader, self.dataset_valid)
            print("EPOCH {} ...".format(e))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        test_accuracy, test_predictions = self.evaluate(self.test_loader, self.dataset_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        
        torch.save(
            {
                'lenet': self.net.state_dict(),
                'opt': self.optimizer.state_dict(),
            },
            ('./models/lenet_classifier.model'),
        )
        print("Model saved")

    def evaluate(self, target_loader, target_dataset):
        predictions = []
        self.net.eval()
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(target_loader):
            output = self.net(images)
            avg_loss += self.criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            predictions.append(pred)

        avg_loss /= len(target_dataset)
        #print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
        accuracy    = float(total_correct) / len(target_dataset)
        return accuracy, np.array(torch.cat(predictions))
        #or if you are in latest Pytorch world
        #return accuracy, np.array(torch.vstack(predictions))


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        self.conv1 = nn.Conv2d(1, 6, (5,5))
        # Layer 2: Convolutional. Output = 10x10x16.
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        self.fc1   = nn.Linear(400, 120)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.fc2   = nn.Linear(120, 84)
        # Layer 5: Fully Connected. Input = 84. Output = 10.
        self.fc3   = nn.Linear(84, 10)
    def forward(self, x):
        # Activation. # Pooling. Input = 28x28x6. Output = 14x14x6.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
         # Activation. # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # Flatten. Input = 5x5x16. Output = 400.
        x = x.flatten(start_dim=1)
        # Activation.
        x = F.relu(self.fc1(x))
        # Activation.
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    img_shape = (32, 32)
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors(y_pred, y_true):
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct_prediction = np.equal(y_pred, y_true)
       
    # Negate the boolean array.
    incorrect = np.equal(correct_prediction, False).bool()
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = X_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = y_pred[incorrect]

    # Get the true classes for those images.
    cls_true = y_true[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9].astype(np.int))
    

if __name__=='main':
    pass