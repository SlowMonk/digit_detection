import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms


import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_var = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

# Step 1: Define the VGG model
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        # Define the architecture based on VGG16
        # You can modify this architecture based on your requirements
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        # 특징 추출 부분 (Feature Extraction)
        self.features = nn.Sequential(
            # 첫 번째 컨볼루션 블록
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 두 번째 컨볼루션 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 세 번째 컨볼루션 블록
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 네 번째 컨볼루션 블록
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 다섯 번째 컨볼루션 블록
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 배치 정규화 추가
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 분류 부분 (Classification)
        self.classifier = nn.Sequential(
            nn.Linear(512 , 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # 특징 추출 네트워크를 통과
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # 분류 네트워크를 통과
        x = self.classifier(x)
        return x


    
class VGG_P(nn.Module):
    def __init__(self, num_classes=11):
        super(VGG, self).__init__()
        # Load the pre-trained VGG16 model from torchvision with pre-trained weights
        self.vgg_model = models.vgg16(pretrained=True)

        # Modify the classifier to match the number of output classes in your dataset
        in_features = self.vgg_model.classifier[6].in_features
        self.vgg_model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg_model(x)
    

# Assume 'device' is defined somewhere as torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load a pre-trained VGG model with pre-trained weights
# class VGG(nn.Module):
#     def __init__(self, num_classes=11):
#         super(VGG, self).__init__()
#         # Load the pre-trained VGG16 model from torchvision with pre-trained weights
#         self.vgg_model = models.vgg16(pretrained=False)

#         # Modify the classifier to match the number of output classes in your dataset
#         in_features = self.vgg_model.classifier[6].in_features
#         self.vgg_model.classifier[6] = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         return self.vgg_model(x)

# Step 2: Define the necessary transformations and DataLoader
# Assuming you have a custom dataset and a proper transformation
# For example, you can use torchvision.transforms.Compose for image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
# ])