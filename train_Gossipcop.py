import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from transformers import BertTokenizer
from dataLoader.data_loader_Gossipcop import load_and_split_data, RumorDataset
from models.models import EnhancedVitBERT


class Args:
    bert_pretrained_dir = "bert-base-uncased"
    vit_pretrained_dir = "pretrained/ViT-B-16.pth"
    model_type = "ViT-B_16"
    img_size = 224
    batch_size = 32
    learning_rate = 2e-5
    num_epochs = 10
    weight_decay = 1e-4


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def evaluate(model, dataloader, device, desc="Validation"):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'vision_features'}
            vision_features = batch['vision_features'].to(device)

            loss = model(
                input_ids=inputs['input_ids'],
                vision_features=vision_features,
                input_mask=inputs['attention_mask'],
                added_attention_mask=None,
                labels=inputs['labels']
            )
            total_loss += loss.item()

            logits = model(
                input_ids=inputs['input_ids'],
                vision_features=vision_features,
                input_mask=inputs['attention_mask'],
                added_attention_mask=None
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(inputs['labels'].cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    return avg_loss, accuracy, precision, recall, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = "datasets/Gossipcop"
    train_df, val_df, test_df = load_and_split_data(dataset_dir)

    tokenizer = BertTokenizer.from_pretrained(Args.bert_pretrained_dir)

    train_dataset = RumorDataset(train_df, tokenizer, img_size=Args.img_size, is_train=True)
    val_dataset = RumorDataset(val_df, tokenizer, img_size=Args.img_size, is_train=False)
    test_dataset = RumorDataset(test_df, tokenizer, img_size=Args.img_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Args.batch_size, shuffle=False)

    model = EnhancedVitBERT(Args, feature_type="full").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Args.learning_rate, weight_decay=Args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    early_stopping = EarlyStopping(patience=3)
    best_f1 = 0

    for epoch in range(Args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Args.num_epochs} [Train]", leave=True):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'vision_features'}
            vision_features = batch['vision_features'].to(device)

            optimizer.zero_grad()
            loss = model(
                input_ids=inputs['input_ids'],
                vision_features=vision_features,
                input_mask=inputs['attention_mask'],
                added_attention_mask=None,
                labels=inputs['labels']
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device, desc="Validation")
        scheduler.step(val_loss)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model_gossipcop.pth")
            print(f"✅ Best model updated at epoch {epoch + 1}")

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {total_loss / len(train_loader):.4f}")
        print(
            f"  Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | P/R/F1: {val_precision:.4f}/{val_recall:.4f}/{val_f1:.4f}")
        print("-" * 60)
        # if early_stopping.step(val_loss):
        #     print("Early stopping triggered!")
        #     break
    print("\n🎯 Evaluating on Test Set with Best Model:")
    model.load_state_dict(torch.load("best_model_gossipcop.pth"))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device, desc="Testing")
    print(
        f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | P/R/F1: {test_precision:.4f}/{test_recall:.4f}/{test_f1:.4f}")


if __name__ == "__main__":
    os.makedirs("pretrained", exist_ok=True)
    main()
