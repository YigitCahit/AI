import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
FILE_PATH = 'veri.txt'
CHUNK_SIZE = 1 * 1024 * 1024 
CHECKPOINT_PATH = 'model_checkpoint.pth'
META_PATH = 'meta.pkl'

print(f"Cihaz: {device}")

if os.path.exists(META_PATH):
    print("Sözlük dosyası bulundu, yükleniyor...")
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
        vocab_size = meta['vocab_size']
else:
    print("Sözlük oluşturuluyor (Büyük dosyalarda bu biraz sürebilir)...")
    unique_chars = set()
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk: break
            unique_chars.update(chunk)
            
    chars = sorted(list(unique_chars))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    with open(META_PATH, 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)
    print(f"Sözlük oluşturuldu! Vocab Size: {vocab_size}")

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

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

model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

if os.path.exists(CHECKPOINT_PATH):
    print("Önceki eğitim bulundu, model yükleniyor...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

batch_size = 32
block_size = 8

def train_on_chunk(text_chunk):
    data = torch.tensor(encode(text_chunk), dtype=torch.long).to(device)
    
    if len(data) <= block_size: return 0
    
    steps_for_chunk = len(data) // batch_size 
    if steps_for_chunk > 100: steps_for_chunk = 100
    
    loss_val = 0
    for _ in range(steps_for_chunk):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        
    return loss_val

print("Dosya parça parça okunuyor ve eğitiliyor...")

total_chunks = 0
try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        while True:
            text_chunk = f.read(CHUNK_SIZE)
            if not text_chunk: break # Dosya bitti
            
            current_loss = train_on_chunk(text_chunk)
            total_chunks += 1
            
            print(f"Parça {total_chunks} tamamlandı. Anlık Loss: {current_loss:.4f}")
            
            if total_chunks % 10 == 0:
                print("--- CHECKPOINT KAYDEDİLİYOR ---")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, CHECKPOINT_PATH)

except KeyboardInterrupt:
    print("\nEğitim kullanıcı tarafından durduruldu.")

print("Eğitim döngüsü tamamlandı/durdu.")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

print("\n--- MODEL ÇIKTISI ---")
print(decode(generate(context, 100)[0].tolist()))