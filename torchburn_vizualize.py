import torch
from torchvision import datasets, transforms
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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



import matplotlib.pyplot as plt
import numpy as np

# HELPERFUNCTION functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



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

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def confuse_model(model):
    y_pred = []
    y_true = []

    # iterate over test data
    for i, (inputs, labels) in enumerate(dataloaders['test']):
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ("DRY", "ICY", "SNOWY", "WET")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *4, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    print(y_pred)
    plt.show()

def main():
    #LOADING THE FULL MODEL
    model = torch.load("C:/Users/Patrick Waldenhofer/Desktop/CAPSTONE/model_full_tf_74%.pth")
    
    confuse_model(model)
    
    # #visualize_model(model)

    # classes = ("DRY", "ICY", "SNOWY", "WET")
    # dataiter = iter(dataloaders['test'])
    # images, labels = dataiter.next()
    # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))



if __name__ == "__main__":
    main()

