import torch

text = "merhaba dünya"
chars = sorted(list(set(text)))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

print(f"Toplam token sayısı: {data.shape[0]}")

n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]