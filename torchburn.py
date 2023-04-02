from sklearn.utils import shuffle

import torch
import torchvision
from torchvision import datasets, transforms

### For imageshow helper function:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# good guide for Torchvision!: https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/
# torchvision documentation for transformation: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html
# good guide for pytorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

batch_size = 4

#"C:/Users/Patrick_Waldenhofer/Desktop/CAPSTONE/Streetsweather_data"

#"C:/Users/Craig/OneDrive - IMC/Semester_4/Machine_Learning/Capstone/Streetsweather_data"
data_dir = "C:/Users/Patrick Waldenhofer/Desktop/CAPSTONE/Streetsweather_data"



# TRANSFORMS ARE NOT FINAL AND TUNED
road_transforms_train = transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.RandomAutocontrast(),
                                       transforms.RandomEqualize(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])
                                    ])

road_transforms_test = transforms.Compose([
                                        transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])
                                    ])

transform_flip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.Resize((224,224)),
    transforms.RandomAutocontrast(),
    transforms.RandomEqualize(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5])
])



#CHECKING CUDA (but does not work)
#checking GPU support
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported

# Cheking other Cuda stuff:
#https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with
#print(torch.cuda.is_available())
#torch.zeros(1).cuda()

train_data = datasets.ImageFolder(data_dir + "/train",  
                                  transform=road_transforms_train)
train_data = train_data + datasets.ImageFolder(data_dir + "/train", transform=transform_flip)
test_data = datasets.ImageFolder(data_dir + "/test",  
                                 transform=road_transforms_test)

trainloader = torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size,shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, 
                                         batch_size=batch_size, shuffle = True)

classes = ("DRY", "ICY", "SNOWY", "WET")

import matplotlib.pyplot as plt
import numpy as np

# HELPERFUNCTION functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

####################################################################################################



import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,4,224,224)
        self.pool = nn.MaxPool2d(4,50,1) #(50,50,32)
        self.conv2 = nn.Conv2d(4,50,1) #(50,50,32)
        self.pool = nn.MaxPool2d(4,50,1) #(25,25,32)
        self.conv3 = nn.Conv2d(50,4,1) #(25,25,64)
        self.pool = nn.MaxPool2d(4,12,1) #(12,12,64)
        self.conv4 = nn.Conv2d(4,100,1) #(100,100,32)
        self.pool = nn.MaxPool2d(2,2,1) #(6,6,128)
        self.fc1 = nn.Linear(100,4)
        self.fc2 = nn.Linear(4, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x

# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN().to(device)
net = CNN()



# Model: "sequential"
# _________________________________________________________________
# Layer (type) Output Shape Param #
# =================================================================
# conv2d (Conv2D) (None, 100, 100, 32) 896

# max_pooling2d (MaxPooling2D (None, 50, 50, 32) 0
# )

# conv2d_1 (Conv2D) (None, 50, 50, 32) 9248

# max_pooling2d_1 (MaxPooling (None, 25, 25, 32) 0
# 2D)

# conv2d_2 (Conv2D) (None, 25, 25, 64) 18496

# max_pooling2d_2 (MaxPooling (None, 12, 12, 64) 0
# 2D)

# conv2d_3 (Conv2D) (None, 12, 12, 128) 73856

# max_pooling2d_3 (MaxPooling (None, 6, 6, 128) 0
# 2D)

# conv2d_4 (Conv2D) (None, 6, 6, 128) 147584

# max_pooling2d_4 (MaxPooling (None, 3, 3, 128) 0
# 2D)

# dropout (Dropout) (None, 3, 3, 128) 0

# global_average_pooling2d (G (None, 128) 0
# lobalAveragePooling2D)

# dense (Dense) (None, 4) 516



# To train a model we need a loss function and an optimizer

import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0
        else:
            #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #running_loss = 0.0
            pass


print("finished training!")

#SAVING MODELS
PATH = './'

# Saving the WHOLE Model!
torch.save(model, os.path.join(PATH,"model_full_last.pth"))

#Saving ONLY the model states!
torch.save(model.state_dict(), os.path.join(PATH,"model_state_last.pth"))

#torch.save(model, "./model_last.pth")
#torch.save(model.state_dict(), PATH)

# SHOW IMAGES

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
imshow(torchvision.utils.make_grid(images))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')