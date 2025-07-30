import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataLoader.data_loader import full_pipeline, WeiboRumorDataset
from models.models import EnhancedVitBERT
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm


class Args:
    bert_pretrained_dir = "bert-base-chinese"
    vit_pretrained_dir = "pretrained/ViT-B-16.pth"
    model_type = "ViT-B_16"
    img_size = 224
    batch_size = 32
    learning_rate = 2e-5
    num_epochs = 10


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
    base_path = "datasets/weibo"
    train_data, val_data, test_data = full_pipeline(
        base_path,
        os.path.join(base_path, "train_id.pickle"),
        os.path.join(base_path, "validate_id.pickle"),
        os.path.join(base_path, "test_id.pickle")
    )
    tokenizer = BertTokenizer.from_pretrained(Args.bert_pretrained_dir)

    train_dataset = WeiboRumorDataset(
        data=train_data, text_tokenizer=tokenizer,
        nonrumor_img_dir=os.path.join(base_path, "nonrumor_images"),
        rumor_img_dir=os.path.join(base_path, "rumor_images"),
        img_size=Args.img_size
    )
    val_dataset = WeiboRumorDataset(
        data=val_data, text_tokenizer=tokenizer,
        nonrumor_img_dir=os.path.join(base_path, "nonrumor_images"),
        rumor_img_dir=os.path.join(base_path, "rumor_images"),
        img_size=Args.img_size
    )
    test_dataset = WeiboRumorDataset(
        data=test_data, text_tokenizer=tokenizer,
        nonrumor_img_dir=os.path.join(base_path, "nonrumor_images"),
        rumor_img_dir=os.path.join(base_path, "rumor_images"),
        img_size=Args.img_size
    )

    train_loader = DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Args.batch_size, shuffle=False)

    # 仅CNN
    # model = EnhancedVitBERT(Args, feature_type="cnn").to(device)

    # 仅Transformer
    # model = EnhancedVitBERT(Args, feature_type="transformer").to(device)

    # 完整模型
    model = EnhancedVitBERT(Args,feature_type="full").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Args.learning_rate)

    print("\n" + "=" * 60)
    print("Initializing Training Process")
    print(f"Model Type: {Args.model_type if hasattr(Args, 'model_type') else 'MultiModal'}")
    print(f"Train/Val/Test Samples: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print("=" * 60 + "\n")

    best_val_f1 = 0
    for epoch in range(Args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Args.num_epochs} [Train]", leave=True)
        for batch in train_pbar:
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
            optimizer.step()

            total_loss += loss.item()

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device, desc="Validation")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model_weibo.pth")
            print(f"✅ Best model updated at epoch {epoch + 1}")

        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Duration: {epoch_duration:.2f}s | Train Loss: {total_loss / len(train_loader):.4f}")
        print(
            f"  Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | P/R/F1: {val_precision:.4f}/{val_recall:.4f}/{val_f1:.4f}")
        print("-" * 60)

    print("\n🎯 Evaluating on Test Set with Best Model:")
    model.load_state_dict(torch.load("best_model_weibo.pth"))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device, desc="Testing")
    print(
        f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | P/R/F1: {test_precision:.4f}/{test_recall:.4f}/{test_f1:.4f}")


if __name__ == "__main__":
    os.makedirs("pretrained", exist_ok=True)
    main()