import torch
import torch.nn as nn
from torch.nn import functional as F
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Eğitim şu cihazda yapılacak: {device.upper()}")
if device == 'cuda':
    print(f"Ekran Kartı: {torch.cuda.get_device_name(0)}")

file_path = 'veri.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

words = text.split()
unique_words = sorted(list(set(words)))
vocab_size = len(unique_words)

stoi = { w:i for i,w in enumerate(unique_words) }
itos = { i:w for i,w in enumerate(unique_words) }

encode = lambda s: [stoi[w] for w in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long).to(device)

print(f"Sözlük Boyutu: {vocab_size}")
print(f"Toplam Token Sayısı: {len(data)}")

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

batch_size = 32
block_size = 2

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

start_time = time.time()
loss_history = []

for step in range(1000):
    
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    optimizer.step()
    
    loss_history.append(loss.item())

    if step % 100 == 0:
        print(f"Adım {step}: Loss = {loss.item():.4f}")

end_time = time.time()
print(f"Eğitim Bitti! Süre: {end_time - start_time:.2f} saniye")
print(f"Başlangıç Loss: {loss_history[0]:.4f}")
print(f"Bitiş Loss: {loss_history[-1]:.4f}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)

def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1) 
        idx_next = torch.multinomial(probs, num_samples=1) 
        idx = torch.cat((idx, idx_next), dim=1) 
    return idx

output = generate(context, max_new_tokens=10)[0].tolist()
print(decode(output))