import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt


data = np.load("mldb.npz")
xs = data["xs"]
N = xs.shape[0]
img_shape = tuple(data["img_shape"])
xs = xs.reshape(N, 1, *img_shape)
ys = data["y"].reshape(-1, 1)
print("SAMPLE SIZE:", N, "IMG.shape", img_shape)
nc = 4
nl = 16
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  nc//2,  5, 3),
            nn.ReLU(),
            nn.Conv2d(nc//2,  nc, 3, 2),
            nn.ReLU(),
            nn.Conv2d(nc,  nc,  3, 2),
            nn.Tanh(),
        )
        self.encoder_dense = nn.Sequential(
            nn.Linear(nc*41*20, nl),
            nn.Tanh(),
        )
        self.decoder_dense = nn.Linear(nl, nc*41*20)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nc,  nc,  3, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nc, nc//2,  3, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=3),
            nn.Conv2d(nc//2,  1,  5, 1),
            nn.Upsample(img_shape)
        )
        self.regressor = nn.Linear(nl, 1)

    def encode(self, X):
        bs = X.shape[0]
        X = self.encoder(X)
        print(X.shape)
        X = X.reshape(bs, -1)
        X = self.encoder_dense(X)
        return X
    def decode(self, X):
        bs = X.shape[0]
        X = self.decoder_dense(X)
        X = X.reshape(bs, nc, 41, 20)
        X = self.decoder(X)
        return X
    def regression(self, X):
        bs = X.shape[0]
        X = self.encoder(X)
        print(X.shape)
        exit()
        X = self.encoder_dense(X)
        X = torch.relu(X)
        X = self.regressor(X)
        return torch.relu(X)


    def forward(self, X):
        X = self.encode(X)
        return self.decode(X)
    
def train():
    model = AutoEncoder()
    model.train()
    mse_loss = nn.MSELoss()
    optim = Adam(params=model.parameters(), lr=1e-3)
    for e in range(10):
        epoch_loss = 0.0
        for x, y in loader:
            bs = x.shape[0]
            optim.zero_grad()
            random_hz_pos = torch.randint(0, img_shape[1], (1,))[0].item()
            x_augment = torch.roll(x, shifts=random_hz_pos, dims=3)
            #xp = model(x_augment)
            #loss = mse_loss(xp, x_augment)
            yp = model.regression(x_augment)
            loss = mse_loss(yp, y)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        print(e, epoch_loss / len(loader))
    return model


loader = DataLoader(TensorDataset(torch.from_numpy(xs).type(torch.float), torch.from_numpy(ys).type(torch.float)), batch_size=32, shuffle=True)
model = train()
model.eval()
X = torch.from_numpy(xs).type(torch.float)
latent = model.encode(X)
plt.scatter(*latent.detach().numpy().T,c=ys[:,0])
plt.savefig("test.png")

