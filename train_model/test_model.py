import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F

# 配置参数
CONFIG = {
    'model': 'resnet34',  # 确保与训练时使用的模型一致
    'model_path': 'checkpoint.pth',  # 保存的模型路径
    'image_size': 224,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomSiameseNetwork(nn.Module):
    def __init__(self, base_model='resnet50', pretrained=False):
        super(CustomSiameseNetwork, self).__init__()
        self.base_model = base_model
        if base_model == 'resnet18':
            self.base_network = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif base_model == 'resnet34':
            self.base_network = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif base_model == 'resnet50':
            self.base_network = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif base_model == 'efficientnet_b0':
            self.base_network = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")

        self.base_network.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )
        self.classifier = nn.Linear(256, 2)

    def forward_one(self, x):
        x = self.base_network(x)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        diff = torch.abs(out1 - out2)
        out = self.classifier(diff)
        return out

class CustomSiameseFeatureExtractor:
    def __init__(self, base_model='resnet50'):
        self.model = CustomSiameseNetwork(base_model=base_model).to(DEVICE)

    def predict(self, img1, img2):
        self.model.eval()
        with torch.no_grad():
            output = self.model(img1, img2)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class.item(), probabilities.squeeze().tolist()

def preprocess_image(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(DEVICE)

def main():
    model_path = CONFIG['model_path']
    extractor = CustomSiameseFeatureExtractor(base_model=CONFIG['model'])
    extractor.model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # 数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 替换为您的两张新图片的路径
    img1_path = r"1300_8.jpg"
    img2_path = r"1300_9.jpg"
    
    img1 = preprocess_image(img1_path, data_transforms)
    img2 = preprocess_image(img2_path, data_transforms)

    predicted_class, probabilities = extractor.predict(img1, img2)
    class_names = ['不相似', '相似']
    print(f'预测类别: {class_names[predicted_class]}')
    print(f'类别概率: 不相似 {probabilities[0]:.4f}, 相似 {probabilities[1]:.4f}')

if __name__ == '__main__':
    print(f"当前设备: {DEVICE}")
    print(f"选择的模型: {CONFIG['model']}")
    print(f"模型路径: {CONFIG['model_path']}")
    main()
