import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler # 믹스드 프리시전
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import time


class TextTransform:
    def __init__(self, vocab, max_len=100):
        self.vocab = vocab
        self.max_len = max_len

    def __call__(self, text):
        text = str(text).lower()
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        
        indices = [self.vocab.get(t, self.vocab["<U NK>"]) for t in tokens]
        
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.vocab["<PAD>"]] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long)


class Vocabulary:
    def __init__(self):
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = 2

    def build(self, texts, max_vocab=10000):
        print("🔨 단어장 구축 중...")
        counter = Counter()
        for text in texts:
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
            counter.update(clean_text.split())
        for word, _ in counter.most_common(max_vocab - 2):
            self.stoi[word] = self.vocab_size
            self.vocab_size += 1

    def get(self, word, default):
        return self.stoi.get(word, default)
    
    def __getitem__(self, word):
        return self.stoi.get(word, self.stoi["<UNK>"])

class IMDBDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.texts = self.df['review'].values
            self.labels = [1 if s == 'positive' else 0 for s in self.df['sentiment'].values]
        else:
            print("⚠️ 가짜 데이터 생성 중...")
            self.texts = ["good movie", "bad movie"] * 50
            self.labels = [1, 0] * 50
            
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.transform:
            text = self.transform(text)
        return text, torch.tensor(label, dtype=torch.float)


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(SentimentRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        out,_ = torch.max(output,dim=1)
        out = self.fc(out)
        return out
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    BATCH_SIZE = 256 
    MAX_LEN = 200    
    EMBED_DIM = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    LR = 0.001

    
    csv_path = r"C:\data\movie review\train.csv"
    
    temp_dataset = IMDBDataset(csv_path) 
    vocab = Vocabulary()
    vocab.build(temp_dataset.texts)
    
    transform = TextTransform(vocab, max_len=MAX_LEN)
    
    full_dataset = IMDBDataset(csv_path, transform=transform)
    train_dataset, valid_dataset = train_test_split(full_dataset, test_size=0.2,random_state=42,shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=False)

    model = SentimentRNN(vocab.vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
    
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler() # Mixed Precision용

    # 저장 경로
    MODEL_DIR = './model_rnn'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    epochs = 30
    start_time = time.time()

    print("🚀 학습 시작!")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)    
            
                    
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        history['loss'].append(train_loss)

        
        model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item() * inputs.size(0)
                
                
                preds = (outputs.view(-1) >= 0).float()
                correct += (preds == labels.view(-1)).sum().item()

        val_loss /= len(valid_loader.dataset)
        val_acc = correct / len(valid_loader.dataset)
        history['val_loss'].append(val_loss)


        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_rnn_model.pth")
            print(f"✅ Best Model Saved! (Acc: {val_acc:.4f})")
            counter = 0
            best_model_time = time.time() - start_time
            best_epoch_idx = epoch + 1
        else:
            counter += 1

        
        if counter >= patience:
            print("Early Stopping!")
            break


    total_time = time.time() - start_time
    print("-" * 50)
    print("Training Finished.")
    print(f"1. Total Training Time : {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"2. Time to Best Model  : {best_model_time // 60:.0f}m {best_model_time % 60:.0f}s (at Epoch {best_epoch_idx})")
    print("-" * 50)
    plt.plot(history['val_loss'], marker='.', c="red", label='Valid Loss')
    plt.plot(history['loss'], marker='.', c="blue", label='Train Loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('IMDB Sentiment Analysis (RNN)')
    plt.show()