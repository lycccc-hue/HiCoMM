import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from transformers import BertTokenizer
from PIL import Image
from io import BytesIO

class RumorDataset(Dataset):
    def __init__(self, data, tokenizer, img_size=224, max_length=128, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_length = max_length

        if is_train:
            self.img_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # 验证集和测试集不做数据增强
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def _load_image_from_bytes(self, img_bytes):
        """
        从 bytes 加载图片并转换为 tensor
        """
        try:
            # 如果 img_bytes 是字典，提取 'bytes' 键的数据
            if isinstance(img_bytes, dict):
                # print("img_bytes 是字典，提取 'bytes' 数据")
                img_bytes = img_bytes.get('bytes', None)
                if img_bytes is None:
                    print("错误：字典中不包含 'bytes' 键")
                    return torch.zeros(3, self.img_size, self.img_size)

            # 确保 img_bytes 现在是 bytes
            if not isinstance(img_bytes, bytes):
                print("错误：img_bytes 仍然不是 bytes，返回空图像")
                return torch.zeros(3, self.img_size, self.img_size)

            img = Image.open(BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return self.img_transform(img)
        except Exception as e:
            print(f"图像加载失败: {e}")
            return torch.zeros(3, self.img_size, self.img_size)  # 失败时返回空图像

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        img_bytes = row['image']
        label = torch.tensor(row['label'], dtype=torch.long)

        # print(f"索引 {idx}: 文本: {text[:30]}... 标签: {label}")

        text_enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        img_tensor = self._load_image_from_bytes(img_bytes)

        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'vision_features': img_tensor,
            'labels': label
        }

def load_and_split_data(parquet_dir):
    file_list = [os.path.join(parquet_dir, file) for file in os.listdir(parquet_dir) if file.endswith(".parquet")]
    df_list = [pd.read_parquet(file, engine="pyarrow") for file in file_list]
    df = pd.concat(df_list, ignore_index=True)

    print("数据集样例:")
    print(df.head())

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # 随机抽取 1800 条作为验证集
    val_df = train_df.sample(n=1800, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("训练集大小:", len(train_df), "验证集大小:", len(val_df), "测试集大小:", len(test_df))

    return train_df, val_df, test_df


# ========== 运行代码示例 ==========

# if __name__ == "__main__":
#     dataset_dir = "../datasets/Gossipcop"
#     train_df, test_df = load_and_split_data(dataset_dir)
#     tokenizer = BertTokenizer.from_pretrained("../bert-base-uncased")
#
#     train_dataset = RumorDataset(train_df, tokenizer)
#     test_dataset = RumorDataset(test_df, tokenizer)
#
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
#     # 打印训练集和测试集中的一条数据（原始数据）
#     print("\n====== 训练集中的第一条数据 ======")
#     print(train_df.iloc[0])
#     print("\n====== 测试集中的第一条数据 ======")
#     print(test_df.iloc[0])
#
#     # 解析并打印第一条数据
#     sample = train_dataset[0]
#     print("\n====== 解析后的训练集样本 ======")
#     print("input_ids:", sample['input_ids'].shape)
#     print("attention_mask:", sample['attention_mask'].shape)
#     print("vision_features:", sample['vision_features'].shape)
#     print("labels:", sample['labels'])
#
#     # 打印原始文本
#     print("\n原始文本内容:", train_df.iloc[0]['text'])
#
#     # 显示图片
#     img = train_dataset._load_image_from_bytes(train_df.iloc[1]['image'])
#     img_pil = transforms.ToPILImage()(img)
#     img_pil.show()
