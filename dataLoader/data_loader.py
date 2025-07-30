import os
import re
import pickle
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from urllib.parse import unquote


def parse_three_lines(lines, label):
    if len(lines) != 3:
        return None
    data_id = lines[0].split('|')[0].strip()
    image_urls = [url.strip() for url in lines[1].split('|') if url.strip()]
    text = lines[2].strip()
    return {'id': data_id, 'text': text, 'images': image_urls, 'label': label}


def load_dataset_by_file(file_path, label):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines_buffer = []
        for line in f:
            lines_buffer.append(line)
            if len(lines_buffer) == 3:
                parsed_data = parse_three_lines(lines_buffer, label)
                if parsed_data:
                    data.append(parsed_data)
                lines_buffer = []
    return data


def load_ids(pickle_path):
    with open(pickle_path, 'rb') as f:
        ids = pickle.load(f)
    return set(str(k) for k in ids.keys())


def split_dataset_by_ids(all_data, train_ids, val_ids, test_ids):
    train_data, val_data, test_data = [], [], []
    for item in all_data:
        if item['id'] in train_ids:
            train_data.append(item)
        elif item['id'] in val_ids:
            val_data.append(item)
        elif item['id'] in test_ids:
            test_data.append(item)
    return train_data, val_data, test_data


def full_pipeline(base_path, train_id_path, val_id_path, test_id_path):
    tweets_path = os.path.join(base_path, 'tweets')
    rumor_data = load_dataset_by_file(os.path.join(tweets_path, 'train_rumor.txt'), 1) + \
                 load_dataset_by_file(os.path.join(tweets_path, 'test_rumor.txt'), 1)
    nonrumor_data = load_dataset_by_file(os.path.join(tweets_path, 'train_nonrumor.txt'), 0) + \
                    load_dataset_by_file(os.path.join(tweets_path, 'test_nonrumor.txt'), 0)

    all_data = rumor_data + nonrumor_data
    train_ids = load_ids(train_id_path)
    val_ids = load_ids(val_id_path)
    test_ids = load_ids(test_id_path)
    return split_dataset_by_ids(all_data, train_ids, val_ids, test_ids)


def sanitize_url_path(url):
    clean_url = unquote(url)
    clean_url = clean_url.replace('\\', '/').replace('//', '/')
    clean_url = re.sub(r'[^a-zA-Z0-9\-_./:]', '_', clean_url)
    filename, ext = os.path.splitext(clean_url)
    if ext.lower() not in ['.jpg', '.jpeg', '.png']:
        clean_url = f"{filename}.jpg"
    return clean_url


class WeiboRumorDataset(Dataset):
    def __init__(self, data, text_tokenizer, nonrumor_img_dir, rumor_img_dir, img_size=224, max_length=128):
        self.data = data
        self.tokenizer = text_tokenizer
        self.nonrumor_img_dir = nonrumor_img_dir
        self.rumor_img_dir = rumor_img_dir
        self.img_size = img_size
        self.max_length = max_length
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _safe_image_path(self, url, label):
        filename = os.path.basename(url)
        clean_name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename.split('?')[0])
        img_dir = self.nonrumor_img_dir if label == 0 else self.rumor_img_dir
        return os.path.join(img_dir, clean_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_enc = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        img_tensor = torch.zeros(3, self.img_size, self.img_size)
        if item['images']:
            for img_url in item['images']:
                local_path = self._safe_image_path(img_url, item['label'])
                if os.path.exists(local_path):
                    try:
                        with Image.open(local_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img_tensor = self.img_transform(img)
                            break
                    except Exception:
                        continue
        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'vision_features': img_tensor,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }
