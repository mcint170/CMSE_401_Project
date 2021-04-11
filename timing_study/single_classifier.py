#### INFO ####
# Code comes from this tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# Code was copied and modified to work on HPCC terminal
##############

## IMPORTS

# Load data loading/manipulating Packages
import torch
import torchvision
import torchvision.transforms as transforms

# Load Neural Network Packages
import torch.nn as nn
import torch.nn.functional as F

# Load Optimizer
import torch.optim as optim

## FUNCTIONS AND CLASSES

# Convolutional Neural Network

class Net(nn.Module):
    """
    Class to crete Convolutional Neural Network for classification.
    that uses nn.Module as base. And redefines forward propagation
    """

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## MAIN FUNCTION

if __name__ == "__main__":
    
    # Check if GPU is avaliable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Calculations done with {}".format(device))

    # Create transform function
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),
                                                        (0.5,0.5,0.5))])
    print("\nLoading in Datasets\n")

    # Load in datasets and loaders
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=1)

    # Create list of possible classifications
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
              'ship', 'truck')
    
    print("\nViewing Random Training Images\n")

    # View some random training images (View training.png)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    torchvision.utils.save_image(images,"training_single.png")

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    print("\nCreating Neural Network\n")

    # Create Neural Net
    net = Net().to(device)

    print("\nCreate Loss function and optimizer\n")

    # Define Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("\nTraining Network\n")
    
    if len(sys.argv) > 1:
        episodes = int(sys.argv[1])
    else:
        episodes = 2

    # Train the network for 2 eopchs
    for epoch in range(episodes):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("\nFinished Training\n")

    print("\nSave trained Model\n")

    # Save trained model
    PATH = './cifar_net_single.pth'
    torch.save(net.state_dict(), PATH)

    print("\nTest the network\n")
    
    print("Test on Some test images")

    # Load some test images
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    torchvision.utils.save_image(images,"testing_single.png")
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Print the predictions for the images above
    outputs = net(images)
    _,predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
          for j in range(4)))

    print("\n Test on all images")

    # Test on all images
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
         100 * correct / total))

    print("\nClass Accuracy\n")

    # Check accuracy of each class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images).to(device)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
              classes[i], 100 * class_correct[i] / class_total[i]))

