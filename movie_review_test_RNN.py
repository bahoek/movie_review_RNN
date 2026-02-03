import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import re
from collections import Counter
import matplotlib.pyplot as plt

class Vocabulary:
    def __init__(self):
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.itos = {0: "<PAD>", 1: "<UNK>"} # 숫자를 단어로 바꾸기 위해 추가 (Decoding용)
        self.vocab_size = 2

    def build(self, texts, max_vocab=10000):
        print("🔨 학습 데이터로 단어장 재구축 중...")
        counter = Counter()
        for text in texts:
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
            counter.update(clean_text.split())
        
        for word, _ in counter.most_common(max_vocab - 2):
            self.stoi[word] = self.vocab_size
            self.itos[self.vocab_size] = word # Reverse mapping 저장
            self.vocab_size += 1

    def get(self, word, default):
        return self.stoi.get(word, default)
    
    def __getitem__(self, word):
        return self.stoi.get(word, self.stoi["<UNK>"])

class TextTransform:
    def __init__(self, vocab, max_len=100):
        self.vocab = vocab
        self.max_len = max_len

    def __call__(self, text):
        text = str(text).lower()
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        
        # <UNK> 처리 수정
        indices = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.vocab["<PAD>"]] * (self.max_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long)

class IMDBDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.texts = self.df['review'].values
            self.labels = [1 if s == 'positive' else 0 for s in self.df['sentiment'].values]
        else:
            print("⚠️ 파일이 없어 더미 데이터를 생성합니다.")
            self.texts = ["good movie", "bad movie"] * 5
            self.labels = [1, 0] * 5
            
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
        # batch_first=True가 학습 코드에 있었으므로 유지
        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        out, _ = torch.max(output, dim=1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ★ 학습 때와 똑같은 하이퍼파라미터 설정
    MAX_LEN = 200    
    EMBED_DIM = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    
    
    train_csv_path = r"C:\data\movie review\train.csv"
    test_csv_path = r"C:\data\movie review\test.csv" # 테스트 파일 경로 (없으면 train으로 테스트)
    model_path = './model_rnn/best_rnn_model.pth'

    if not os.path.exists(test_csv_path):
        print(f"⚠️ 테스트 파일({test_csv_path})이 없어 학습 파일로 대체합니다.")
        test_csv_path = train_csv_path

    # 1. 단어장(Vocabulary) 복구
    # 학습 때 사용한 단어 사전을 그대로 복구해야 모델이 단어를 알아듣습니다.
    temp_dataset = IMDBDataset(train_csv_path) 
    vocab = Vocabulary()
    vocab.build(temp_dataset.texts) # 학습 데이터로 빌드!
    
    transform = TextTransform(vocab, max_len=MAX_LEN)

    # 2. 테스트 데이터셋 로드
    print("📁 테스트 데이터를 불러오는 중...")
    test_dataset = IMDBDataset(test_csv_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0) # Windows에선 workers 0 권장 (에러 시)

    # 3. 모델 로드
    print("🧠 모델을 로드하는 중...")
    model = SentimentRNN(vocab.vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ 학습된 RNN 모델 로드 완료!")
    else:
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        exit()

    # 4. 전체 정확도 평가
    print("🚀 정확도 측정 시작...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Sigmoid를 통과시켜 확률로 변환 (0~1)
            probs = torch.sigmoid(outputs.view(-1))
            predicted = (probs >= 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()

    accuracy = 100 * correct / total
    print(f"\n🏆 최종 테스트 정확도: {accuracy:.2f}%")

    # ==========================================
    # [3] 결과 눈으로 확인 (시각화 대체)
    # ==========================================
    print("\n👀 예측 결과 샘플 확인 (상위 5개)")
    print("=" * 80)
    
    dataiter = iter(test_loader)
    inputs, labels = next(dataiter)
    inputs = inputs.to(device)
    
    # 예측 수행
    outputs = model(inputs)
    probs = torch.sigmoid(outputs.view(-1))
    predicted = (probs >= 0.5).float()

    # 텍스트 복원 함수 (Index -> Word)
    def decode_text(indices, vocab):
        tokens = []
        for idx in indices:
            idx = idx.item()
            if idx == vocab.stoi["<PAD>"]: continue # 패딩은 무시
            tokens.append(vocab.itos.get(idx, "<UNK>"))
        return " ".join(tokens)

    # 5개만 출력
    for i in range(5):
        raw_text = decode_text(inputs[i], vocab)
        pred_label = "Positive" if predicted[i].item() == 1 else "Negative"
        act_label = "Positive" if labels[i].item() == 1 else "Negative"
        prob_val = probs[i].item() * 100
        
        # 맞으면 파란색(혹은 O), 틀리면 빨간색(혹은 X) 표시 (터미널 환경에 따라 색상 지원 다를 수 있음)
        result_mark = "✅" if predicted[i] == labels[i] else "❌"
        
        print(f"Sample {i+1} {result_mark}")
        print(f"📝 Text : {raw_text[:100]}...") # 너무 기니까 100자만
        print(f"📊 Pred : {pred_label} ({prob_val:.1f}%)")
        print(f"🏷️ Real : {act_label}")
        print("-" * 80)