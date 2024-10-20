import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import random

# 配置参数
CONFIG = {
    'model': 'resnet50',  # 可选: 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
    'data_dir': r'C:\Users\Runker\Desktop\train',  # 请替换为您的数据目录路径
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'seed': 42,
    'freeze_layers': True,  # 是否冻结预训练模型的大部分层
    'unfreeze_layers': 2  # 解冻最后几层进行微调
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])

class SimilarityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        for label, category in enumerate(['yes', 'no']):
            category_path = os.path.join(root_dir, category)
            for id_folder in os.listdir(category_path):
                id_path = os.path.join(category_path, id_folder)
                if os.path.isdir(id_path):
                    images = [f for f in os.listdir(id_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if len(images) == 2:
                        self.samples.append((os.path.join(id_path, images[0]), 
                                             os.path.join(id_path, images[1])))
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path = self.samples[idx]
        label = self.labels[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.long)

class CustomSiameseNetwork(nn.Module):
    def __init__(self, base_model='resnet50', pretrained=True):
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

        # 移除最后的全连接层
        self.base_network = nn.Sequential(*list(self.base_network.children())[:-1])
        
        # 添加新的层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(256, 2)

        # 冻结预训练模型的层
        if CONFIG['freeze_layers']:
            for param in self.base_network.parameters():
                param.requires_grad = False
            
            # 解冻最后几层
            for i, (name, param) in enumerate(reversed(list(self.base_network.named_parameters()))):
                if i < CONFIG['unfreeze_layers'] * 2:  # *2 是因为每层通常有权重和偏置
                    param.requires_grad = True
                else:
                    break

    def forward_one(self, x):
        x = self.base_network(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        diff = torch.abs(out1 - out2)
        out = self.classifier(diff)
        return out

class CustomSiameseFeatureExtractor:
    def __init__(self, base_model='resnet50', lr=1e-4, weight_decay=1e-5):
        self.model = CustomSiameseNetwork(base_model=base_model).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        
        # 只优化需要梯度的参数
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

    def train(self, train_loader, val_loader, num_epochs):
        writer = SummaryWriter()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (img1, img2, label) in enumerate(train_loader):
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(img1, img2)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'轮次 {epoch}, 批次 {batch_idx}, 损失: {loss.item()}')
            
            val_loss, accuracy = self.validate(val_loader)
            
            self.scheduler.step()
            
            writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
            
            print(f'轮次 {epoch}, 训练损失: {train_loss/len(train_loader):.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {accuracy:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("保存最佳模型")
        
        writer.close()

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                output = self.model(img1, img2)
                loss = self.criterion(output, label)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
        accuracy = correct / total
        return val_loss / len(val_loader), accuracy

def get_data_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    data_transforms = get_data_transforms()

    # 创建数据集
    dataset = SimilarityDataset(root_dir=CONFIG['data_dir'], transform=data_transforms)

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 初始化和训练模型
    extractor = CustomSiameseFeatureExtractor(base_model=CONFIG['model'], lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    extractor.train(train_loader, val_loader, num_epochs=CONFIG['epochs'])

    # 在测试集上评估模型
    extractor.model.load_state_dict(torch.load('best_model.pth'))
    test_loss, accuracy = extractor.validate(test_loader)
    print(f'测试集损失: {test_loss:.4f}, 测试集准确率: {accuracy:.4f}')

if __name__ == '__main__':
    print(f"当前设备: {DEVICE}")
    print(f"选择的模型: {CONFIG['model']}")
    print(f"数据集目录: {CONFIG['data_dir']}")
    print(f"批次大小: {CONFIG['batch_size']}")
    print(f"训练轮数: {CONFIG['epochs']}")
    print(f"学习率: {CONFIG['learning_rate']}")
    print(f"权重衰减: {CONFIG['weight_decay']}")
    print(f"是否冻结层: {CONFIG['freeze_layers']}")
    print(f"解冻层数: {CONFIG['unfreeze_layers']}")

    main()