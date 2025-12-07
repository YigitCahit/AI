import torch
import pickle
from model import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'model_checkpoint.pth'
META_PATH = 'meta.pkl'

print("Sözlük yükleniyor...")
with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

model = GPTLanguageModel().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model sohbete hazır!")

while True:
    start_text = input("\nSen: ")
    if start_text.lower() == "çıkış": break
    
    context_idxs = torch.tensor(encode(start_text), dtype=torch.long, device=device).unsqueeze(0)
    generated_text = start_text
    
    with torch.no_grad():
            idx_cond = context_idxs[:, -512:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            context_idxs = torch.cat((context_idxs, idx_next), dim=1)
            generated_text += decode([idx_next.item()])
            
            print(decode([idx_next.item()]), end='', flush=True)