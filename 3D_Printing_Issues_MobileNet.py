import torch
from torchvision import datasets, transforms, models
import pandas as pd
import os
from PIL import Image
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision.models.densenet import DenseNet121_Weights

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
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

def main():
  
    model = models.mobilenet_v2(pretrained=True)
    num_classes = 2

    # model.classifier = nn.Sequential(
    # nn.Dropout(0.60),
    # nn.Linear(model.classifier.in_features, num_classes)
    # )

    model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, num_classes)
)


    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 6
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
    submission_df.to_csv('submission_mobilenet.csv', index=False)

if __name__ == "__main__":
    main()
    

