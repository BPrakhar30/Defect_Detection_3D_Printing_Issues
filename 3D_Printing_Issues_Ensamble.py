import torch
from torchvision import datasets, transforms, models
import pandas as pd
import os
from PIL import Image
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision.models.resnet import ResNet, BasicBlock

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)
        image = self.transform(image)

        if self.mode == "train":
            label = self.data.iloc[idx, 3]
            return image, label
        else:
            return image
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
train_csv = ImageDataset(r"/home/prakharb/3D_Printing/images/train.csv", r"/home/prakharb/3D_Printing/images/training_data" ,transform = transform)
train_loader = torch.utils.data.DataLoader(train_csv, batch_size = 1, shuffle = True)
test_dataset = ImageDataset(r"/home/prakharb/3D_Printing/images/test.csv", r"/home/prakharb/3D_Printing/images/testing_data" , transform = transform, mode = "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle=False, num_workers = 0)

class CustomResNet(ResNet):
    def __init__(self, block, layers, num_classes, dropout_p=0.5):
        super(CustomResNet, self).__init__(block, layers)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.fc.in_features, num_classes)
        )

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# def custom_resnet18(pretrained=True, num_classes=2, dropout_p=0.5):
#     model = CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes, dropout_p)
#     if pretrained:
#         state_dict = models.resnet18(pretrained=True).state_dict()
#         model.load_state_dict(state_dict, strict=False)
#     return model

def custom_resnet(pretrained=True, num_classes=2, dropout_p=0.5, arch='resnet18'):
    model = CustomResNet(BasicBlock, [2, 2, 2, 2] if arch == 'resnet18' else [3, 4, 6, 3], num_classes, dropout_p)
    if pretrained:
        state_dict = getattr(models, arch)(pretrained=True).state_dict()
        model.load_state_dict(state_dict, strict=False)
    return model


def main():
    # Train and evaluate multiple models
    models_archs = ['resnet18', 'resnet34']
    predictions_list = []

    for arch in models_archs:
        model = custom_resnet(pretrained=True, num_classes=2, dropout_p=0.5, arch=arch)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        epochs = 20

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

            print("Model: {}, Epoch: {}, Loss: {:.4f}".format(arch, epoch + 1, loss.item()))

        model.eval()
        predictions = []

        for images in test_loader:
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy().tolist())

        predictions_list.append(predictions)

    # Average predictions from multiple models
    avg_predictions = [round(sum(x) / len(x)) for x in zip(*predictions_list)]

    submission_df = pd.DataFrame({'img_path': test_dataset.data['img_path'], 'has_under_extrusion': avg_predictions})
    submission_df.to_csv('submission_ensamble.csv', index=False)

if __name__ == "__main__":
    main()
