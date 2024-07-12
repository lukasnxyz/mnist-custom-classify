import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import trange

d_opts = [('cuda', torch.cuda.is_available()), ('mps', torch.backends.mps.is_available()), ('cpu', True)]
device = next(device for device, available in d_opts if available)
print(f'using device: {device}')

# load data
def load_data(dir_name: str):
    data = list()
    with open(dir_name, 'r') as f:
        [data.append(np.array(line.split(','), dtype=np.float32)) for line in f]
    data = np.asarray(data)
    return torch.from_numpy(data)
    
data = load_data("data/mnist_data.csv")
# split into train/val sets
n = int(0.9*len(data))
tr_data, val_data = data[:n], data[n:]
# split into X,Y
Xtr, Ytr = tr_data[:, 1:], tr_data[:, 0]
Xval, Yval = val_data[:, 1:], val_data[:, 0]
# rescale 0-255 to 0.0-1.0
Xtr = Xtr/Xtr.max()
Xval = Xval/Xval.max()
# move data to device
Xtr, Ytr = Xtr.to(device), Ytr.to(device)
Xval, Yval = Xval.to(device), Yval.to(device)

# hyperparameters
torch.manual_seed(42)
epochs = 10000
epoch_itr = 250
batch_size = 32
learning_rate = 1e-4

class mnist_model(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
        )
        self.block2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
        )
        self.block3 = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

m = mnist_model(n_in=Xtr.shape[1], n_hidden=128, n_out=10).to(device)
print(f'num of params: {sum([p.numel() for p in m.parameters()])}')
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

m.train()
start_time = time.time()
for epoch in (t := trange(epochs)):
    # mini batching
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = m.forward(Xb)
    loss = loss_fn(logits, Yb)

    optimizer.zero_grad()
    loss.backward()

    learning_rate = 1e-1 if epoch > 5000 else 1e-4
    optimizer.step()
    
    # stats
    if epoch % epoch_itr == 0:
        t.set_description(f'loss {loss.item():.4f}')
    
end_time = time.time()
print(f'time to train {end_time - start_time:.1f}s')

@torch.no_grad()
def split_loss(split: str):
    x,y = {
        'train': (Xtr, Ytr),
        'val': (Xval, Yval),
    }[split]
    logits = m(x.float())
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

with torch.inference_mode():
    split_loss('train')
    split_loss('val')

m.eval()
with torch.inference_mode():
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc
    logits = m(Xval)
    acc = accuracy_fn(y_pred=logits.argmax(dim=1),
                     y_true=Yval)
    print(f'Accuracy: {acc:.2f}%')