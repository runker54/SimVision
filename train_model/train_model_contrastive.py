import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import GradScaler, autocast

CONFIG = {
    'model': 'resnet18',
    'data_dir': r'C:\Users\Runker\Desktop\train',
    'batch_size': 8,
    'epochs': 200,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'seed': 42,
    'margin': 1.0  # 对比损失的边界值
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

class SiameseNetwork(nn.Module):
    def __init__(self, base_model='resnet18'):
        super(SiameseNetwork, self).__init__()
        if base_model == 'resnet18':
            self.base_network = models.resnet18(pretrained=True)
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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class SiameseNetworkTrainer:
    def __init__(self, base_model='resnet18', lr=1e-4, weight_decay=1e-4, margin=1.0):
        self.model = SiameseNetwork(base_model=base_model).to(DEVICE)
        self.criterion = ContrastiveLoss(margin=margin)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CONFIG['epochs'])
        self.scaler = GradScaler()

    def train(self, train_loader, val_loader, num_epochs):
        writer = SummaryWriter()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (img1, img2, label) in enumerate(train_loader):
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE).float()
                self.optimizer.zero_grad()
                
                with autocast():
                    output1, output2 = self.model(img1, img2)
                    loss = self.criterion(output1, output2, label)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'轮次 {epoch}, 批次 {batch_idx}, 损失: {loss.item()}')
            
            val_loss = self.validate(val_loader)
            
            self.scheduler.step()
            
            writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            print(f'轮次 {epoch}, 训练损失: {train_loss/len(train_loader):.4f}, 验证损失: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("保存最佳模型")
        
        writer.close()

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE).float()
                output1, output2 = self.model(img1, img2)
                loss = self.criterion(output1, output2, label)
                val_loss += loss.item()
        return val_loss / len(val_loader)

class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        for t in self.transforms:
            if random.random() < self.p:
                img = t(img)
        return img

def get_data_transforms():
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            color_jitter,
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    data_transforms = get_data_transforms()
    dataset = SimilarityDataset(root_dir=CONFIG['data_dir'], transform=data_transforms)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    trainer = SiameseNetworkTrainer(base_model=CONFIG['model'], lr=CONFIG['learning_rate'], 
                                    weight_decay=CONFIG['weight_decay'], margin=CONFIG['margin'])
    trainer.train(train_loader, val_loader, num_epochs=CONFIG['epochs'])

    # 测试
    trainer.model.load_state_dict(torch.load('best_model.pth'))
    test_loss = trainer.validate(test_loader)
    print(f'测试集损失: {test_loss:.4f}')

if __name__ == '__main__':
    print(f"当前设备: {DEVICE}")
    print(f"选择的模型: {CONFIG['model']}")
    print(f"数据集目录: {CONFIG['data_dir']}")
    print(f"批次大小: {CONFIG['batch_size']}")
    print(f"训练轮数: {CONFIG['epochs']}")
    print(f"学习率: {CONFIG['learning_rate']}")
    print(f"权重衰减: {CONFIG['weight_decay']}")
    print(f"对比损失边界值: {CONFIG['margin']}")

    main()

