import torch
import torch.nn as nn

text = "merhaba dünya"

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

print(f"Toplam token sayısı: {data.shape[0]}")

n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

token_embedding_table = nn.Embedding(vocab_size, vocab_size)

print(f"Tablo (Weight) boyutu: {token_embedding_table.weight.shape}")

idx = data[:1]

print(f"\nGirdi Harf ID'si: {idx}") 
print(f"Bu ID'nin harf karşılığı: '{decode(idx.tolist())}'")

logits = token_embedding_table(idx)

print(f"Çıktı Puanları (Logits) Boyutu: {logits.shape}")
print(f"Açıklama: [1 (Girdi Sayısı), {vocab_size} (Olası her harf için puan)]")
print(f"Üretilen ham puanlar (İlk 5 tanesi): {logits[0, :5]}")