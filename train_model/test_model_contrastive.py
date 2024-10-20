import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F

CONFIG = {
    'model': 'resnet18',
    'model_path': 'best_model.pth',
    'image_size': 224,
    'similarity_threshold': 0.5  # 相似性阈值
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(nn.Module):
    def __init__(self, base_model='resnet18'):
        super(SiameseNetwork, self).__init__()
        if base_model == 'resnet18':
            self.base_network = models.resnet18(pretrained=False)
            self.feature_dim = 512
        else:
            raise ValueError(f"不支持的基础模型: {base_model}")

        self.base_network = nn.Sequential(*list(self.base_network.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_one(self, x):
        x = self.base_network(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

class SiameseNetworkTester:
    def __init__(self, base_model='resnet18'):
        self.model = SiameseNetwork(base_model=base_model).to(DEVICE)

    def predict(self, img1, img2):
        self.model.eval()
        with torch.no_grad():
            output1, output2 = self.model(img1, img2)
            distance = F.pairwise_distance(output1, output2)
            similarity = 1 / (1 + distance.item())  # 将距离转换为相似度
        return similarity

def preprocess_image(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(DEVICE)

def main():
    model_path = CONFIG['model_path']
    tester = SiameseNetworkTester(base_model=CONFIG['model'])
    tester.model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    data_transforms = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img1_path = r"1300_1.jpg"
    img2_path = r"1300_2.jpg"
    
    img1 = preprocess_image(img1_path, data_transforms)
    img2 = preprocess_image(img2_path, data_transforms)

    similarity = tester.predict(img1, img2)
    print(f'相似度: {similarity:.4f}')
    
    if similarity > CONFIG['similarity_threshold']:
        print("预测结果: 相似")
    else:
        print("预测结果: 不相似")

if __name__ == '__main__':
    print(f"当前设备: {DEVICE}")
    print(f"选择的模型: {CONFIG['model']}")
    print(f"模型路径: {CONFIG['model_path']}")
    main()
