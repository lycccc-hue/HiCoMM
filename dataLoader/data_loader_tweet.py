import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import random
from torch.utils.data import DataLoader, Subset


class TweetDataset(Dataset):
    def __init__(self, txt_file, img_dir, tokenizer, img_size=224, max_length=128, transform=None):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 标签映射表
        self.label_map = {'fake': 1, 'real': 0}

        self.data = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过首行表头
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    tweet_id, text, user_id, image_id, username, timestamp, label_str = parts
                    label_str = label_str.strip().lower()  # 去掉空格和大小写
                    if label_str in self.label_map:
                        label = self.label_map[label_str]  # 映射成 0 或 1
                        self.data.append({
                            'tweet_id': tweet_id,
                            'text': text,
                            'image_id': image_id,
                            'label': label
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 文本编码
        text_enc = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 图像路径
        img_path = os.path.join(self.img_dir, f"{item['image_id']}.jpg")

        # 图像加载和转换
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            vision_features = self.transform(image)
        else:
            vision_features = torch.zeros(3, self.img_size, self.img_size)  # 缺失图像处理

        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'vision_features': vision_features,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


# 数据加载器，包含训练、验证、测试集划分
def get_dataloaders(txt_file, img_dir, tokenizer, batch_size=32, train_ratio=0.7, val_ratio=0.15,
                                 shuffle=True, seed=42):
    dataset = TweetDataset(txt_file, img_dir, tokenizer)
    total_size = len(dataset)

    # 生成索引列表
    indices = list(range(total_size))

    # 是否随机打乱
    if shuffle:
        random.seed(seed)  # 固定种子，保证可复现
        random.shuffle(indices)

    # 手动划分索引
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # 用 Subset 包装对应子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
