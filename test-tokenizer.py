import torch
import torch.nn as nn

text = "merhaba dünya bugün hava çok güzel merhaba dünya"

words = text.split() 
unique_words = sorted(list(set(words)))
vocab_size = len(unique_words)

stoi = { w:i for i,w in enumerate(unique_words) }
itos = { i:w for i,w in enumerate(unique_words) }

encode = lambda s: [stoi[w] for w in s.split()] 
decode = lambda l: ' '.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

print(f"Toplam token (kelime) sayısı: {data.shape[0]}")
print(f"Sözlükteki benzersiz kelimeler: {unique_words}")

n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

token_embedding_table = nn.Embedding(vocab_size, vocab_size)

print(f"Tablo (Weight) boyutu: {token_embedding_table.weight.shape}")

idx = data[:1]

print(f"\nGirdi Kelime ID'si: {idx}") 
print(f"Bu ID'nin kelime karşılığı: '{decode(idx.tolist())}'")

logits = token_embedding_table(idx)

print(f"Çıktı Puanları (Logits) Boyutu: {logits.shape}")

scores = logits[0]

highest_index = torch.argmax(scores).item()
print(f"Modelin Tahmini (En yüksek puan): '{itos[highest_index]}'")