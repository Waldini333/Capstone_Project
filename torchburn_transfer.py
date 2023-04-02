from sklearn.utils import shuffle

import torch
import torchvision
from torchvision import datasets, transforms, models

### For imageshow helper function:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.RandomAutocontrast(),
                                       transforms.RandomEqualize(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])
                                    ]),
    'test': transforms.Compose([
                                        transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])
                                    ]),
}

data_dir = "C:/Users/Patrick Waldenhofer/Desktop/CAPSTONE/Streetsweather_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


import matplotlib.pyplot as plt
import numpy as np


# HELPERFUNCTION functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

###########################!!!!!!!!!!!!!!!!!!!!!!!!!!'''''########################################
import time
import copy
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

#torch.multiprocessing.freeze_support()
#print("freeze support loaded")

#CNN
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



###############################################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():
    #model_ft.load_state_dict(torch.load("C:/Users/Patrick Waldenhofer/Desktop/CAPSTONE/model_last.pth"))


    #LOADING THE FULL MODEL
    #model_ft = torch.load("C:/Users/Patrick Waldenhofer/Desktop/CAPSTONE/model_full_aug_55%.pth")

    model_ft = models.resnet18(pretrained=True)
    #model_ft = models.vgg16(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    #num_ftrs = model_ft.axial.model.classifier[0].in_features

    #num_ftrs = model_ft.axial.model.classifier[0].in_features
    #num_ftrs = model_ft.fc3.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    #model_ft.axial.model.classifier[0] = nn.Linear(num_ftrs, 4)
    
    
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    #model_ft = CNN()
    #model_ft.load_state_dict(model_ft_state_dict)

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    

    #visualize_model(model_ft,)


    model_ft = train_model(model_ft,criterion,optimizer_ft,scheduler,num_epochs=25)

    # Saving the WHOLE Model!
    PATH = './'
    torch.save(model_ft, os.path.join(PATH,"model_full_tf_last.pth"))


if __name__ == "__main__":
    main()

#print("finished training!")

# #SAVING MODELS
# PATH = './'
# torch.save(os.path.join(PATH,"model_last.pth"))
# #torch.save(model.state_dict(), PATH)

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                               for j in range(4)))


# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')