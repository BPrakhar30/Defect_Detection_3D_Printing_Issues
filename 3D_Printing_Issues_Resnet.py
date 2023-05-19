import torch
from torchvision import datasets, transforms
import pandas as pd
import os
from PIL import Image
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform = transform, mode="train"):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0].replace("/", "\\"))
        image = Image.open(img_path)
        image = self.transform(image)

        if self.mode == "train":
            label = self.data.iloc[idx, 3]
            return image, label
        else:
            return image
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

train_csv = ImageDataset(r"C:\Users\prakh\Desktop\CMU_EDU\Sem2\3dprintingissues\early-detection-of-3d-printing-issues\images\train.csv", r"C:\Users\prakh\Desktop\CMU_EDU\Sem2\3dprintingissues\early-detection-of-3d-printing-issues\images\training_data", transform = transform)
train_loader = torch.utils.data.DataLoader(train_csv, batch_size = 32, shuffle = True)

test_dataset = ImageDataset(r"C:\Users\prakh\Desktop\CMU_EDU\Sem2\3dprintingissues\early-detection-of-3d-printing-issues\images\test.csv", r"C:\Users\prakh\Desktop\CMU_EDU\Sem2\3dprintingissues\early-detection-of-3d-printing-issues\images\testing_data" , transform = transform, mode = "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle=False, num_workers = 0)

# -*- coding: utf-8 -*-

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample = None, stride = 1, l2_reg=1e-5
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            bias = False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
        self.l2_reg = l2_reg

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
    def l2_loss(self):
        loss = 0.0
        for param in self.parameters():
            loss += 0.5 * self.l2_reg * torch.norm(param)**2
        return loss


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, l2_reg=1e-5):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1, l2_reg=l2_reg
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2, l2_reg=l2_reg
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2, l2_reg=l2_reg
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2, l2_reg=l2_reg
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout1(x)
        x = self.layer3(x)
        x = self.dropout2(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride, l2_reg):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


##################### for test 22 - 514 & 22 - 024

def main():

    def ResNet50(img_channel=3, num_classes=1000):
        return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

    def ResNet101(img_channel=3, num_classes=1000):
        return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)

    def ResNet152(img_channel=3, num_classes=1000):
        return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)
    
    model = ResNet50()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 1
    for epoch in range(epochs):
        for images, labels in train_loader:
            if images is None or labels is None:
                print("Skipping batch due to None values.")
                continue

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        scheduler.step()
        
        print("Epoch: {}, Loss: {:.4f}".format(epoch+1, loss.item()))

    model.eval()
    predictions = []

    for images in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy().tolist())
    submission_df = pd.DataFrame({'img_path': test_dataset.data['img_path'], 'has_under_extrusion': predictions})
    submission_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
    

