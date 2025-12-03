import torch
import torch.nn as nn
from torch.nn import functional as F

text = "merhaba dünya bugün hava çok güzel merhaba dünya"

words = text.split() 
unique_words = sorted(list(set(words)))
vocab_size = len(unique_words)

stoi = { w:i for i,w in enumerate(unique_words) }
itos = { i:w for i,w in enumerate(unique_words) }

encode = lambda s: [stoi[w] for w in s.split()] 
decode = lambda l: ' '.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

print(f"Sözlük Boyutu: {vocab_size}")

token_embedding_table = nn.Embedding(vocab_size, vocab_size)

idx = data[:1] 
print(f"Girdi: '{decode(idx.tolist())}' (ID: {idx.item()})")

targets = data[1:2] 
print(f"Olması Gereken (Hedef): '{decode(targets.tolist())}' (ID: {targets.item()})")

logits = token_embedding_table(idx)

loss = F.cross_entropy(logits, targets)

print(f"Tahmin Puanları (Logits): {logits}")
print(f"Hesaplanan Loss (Hata) Değeri: {loss.item():.4f}")

correct_answer_id = targets.item()
correct_answer_score = logits[0, correct_answer_id].item()

print(f"Modelin doğru cevaba ('{decode([correct_answer_id])}') verdiği puan: {correct_answer_score:.4f}")