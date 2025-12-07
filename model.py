import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
FILE_PATH = 'veri.txt'
CHUNK_SIZE = 1048576 * 10
CHECKPOINT_PATH = 'model_checkpoint.pth'
META_PATH = 'meta.pkl'

batch_size = 48
block_size = 256

n_embd = 640
n_head = 8
n_layer = 10
dropout = 0.0
learning_rate = 2.3e-4

NUM_EPOCHS = 100

print(f"Cihaz: {device}")

if os.path.exists(META_PATH):
    print("Sözlük yükleniyor...")
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
        vocab_size = meta['vocab_size']
else:
    print("Sözlük oluşturuluyor...")
    unique_chars = set()
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk: break
                unique_chars.update(chunk)
    except FileNotFoundError:
        print("HATA: veri.txt yok!")
        exit()
            
    chars = sorted(list(unique_chars))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    with open(META_PATH, 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)
    print(f"Sözlük tamam. Vocab: {vocab_size}")

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if os.path.exists(CHECKPOINT_PATH):
    print("Kayıt dosyası bulundu, yükleniyor...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model yüklendi.")
    except:
        print("Eski model uyumsuz, sıfırdan başlanıyor.")

def train_on_chunk(text_chunk):
    data = torch.tensor(encode(text_chunk), dtype=torch.long).to(device)
    if len(data) <= block_size: return 0

    steps_for_chunk = 50 
    
    loss_val = 0
    
    sys.stdout.write(f"   [Eğitiliyor: {steps_for_chunk} Adım] -> ")
    
    for i in range(steps_for_chunk):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        
        sys.stdout.write("█") 
        sys.stdout.flush()
        
    print(f" Bitti. Loss: {loss_val:.4f}")
    torch.cuda.empty_cache()
    return loss_val

def generate_sample():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    for _ in range(50): 
        idx_cond = context[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
    return decode(context[0].tolist())

print("--- EĞİTİM BAŞLIYOR ---")
print(f"Model Boyutu: {sum(p.numel() for p in model.parameters())/1e6:.2f} Milyon Parametre")

total_chunks = 0
try:
    for epoch in range(NUM_EPOCHS):
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            while True:
                text_chunk = f.read(CHUNK_SIZE)
                if not text_chunk: break 
                
                total_chunks += 1
                print(f"\nParça {total_chunks} Okundu:")
                current_loss = train_on_chunk(text_chunk)
                
                if total_chunks % 1 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, CHECKPOINT_PATH)
                    
                    msg = generate_sample()
                    print(f"\033[93m>> Model:\033[0m {msg}")

except KeyboardInterrupt:
    print("\nDurdu.")