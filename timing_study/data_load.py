# Import required packages
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":
    # Load in data to make calculation timing fair
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

