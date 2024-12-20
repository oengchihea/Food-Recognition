import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels//8, in_channels, kernel_size=1)
        
    def forward(self, x):
        attention = F.avg_pool2d(x, x.size()[2:])
        attention = F.relu(self.conv1(attention))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention

class FoodClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnetv2_rw_s'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        
        # Get the number of features from the last layer
        dummy_input = torch.randn(1, 3, 224, 224)
        features = self.backbone(dummy_input)
        last_channel = features[-1].shape[1]
        
        self.attention = AttentionModule(last_channel)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(last_channel, 1024)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Texture-specific branch
        self.texture_conv = nn.Sequential(
            nn.Conv2d(last_channel, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.texture_pool = nn.AdaptiveAvgPool2d(1)
        self.texture_fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Extract features through backbone
        features = self.backbone(x)
        x = features[-1]  # Use the last feature map
        
        # Apply attention
        x = self.attention(x)
        
        # Main classification branch
        main = self.global_pool(x)
        main = main.view(main.size(0), -1)
        main = self.dropout1(main)
        main = F.relu(self.fc1(main))
        main = self.dropout2(main)
        main = self.fc2(main)
        
        # Texture-specific branch
        texture = self.texture_conv(x)
        texture = self.texture_pool(texture)
        texture = texture.view(texture.size(0), -1)
        texture = self.texture_fc(texture)
        
        # Combine predictions
        return main + 0.5 * texture

