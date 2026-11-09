"""Linear probe training (the recipe of the paper's probes).

Inputs are mean-centred with the float64 mean of the training representations (the same mean is subtracted from the
test set). A probe is a single nn.Linear layer trained with Adam (lr 1e-3, weight decay 1e-4 on weights and bias),
batch 256, 10 epochs; the final epoch is kept. Classification probes minimise cross-entropy, regression probes (one
output) the mean squared error. The probe seed fixes both the initialisation and the shuffling order.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LR, WEIGHT_DECAY, BATCH_SIZE, EPOCHS = 1e-3, 1e-4, 256, 10


def centre(train, test):
    mean = train.double().mean(0)
    return (train.double() - mean).float(), (test.double() - mean).float()


class _Indices(Dataset):
    """A DataLoader over this dataset yields each shuffled batch of indices as one tensor."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitems__(self, idx):
        return torch.tensor(idx, dtype=torch.long)


def shuffled_batches(n, batch_size, generator):
    """A function returning one epoch of shuffled index batches. It draws from `generator` exactly as
    DataLoader(range(n), batch_size, shuffle=True, generator=generator) does, so the batches are the same."""
    loader = DataLoader(_Indices(n), batch_size=batch_size, shuffle=True, generator=generator, collate_fn=lambda b: b)
    return lambda: list(loader)


def init_linear(d_in, d_out, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.nn.Linear(d_in, d_out)


def train_probe(X, y, n_classes, seed, device, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE):
    """Returns the trained nn.Linear (on `device`). X float32 [n, d], y int64 [n]."""
    probe = init_linear(X.shape[1], n_classes, seed).to(device)
    epoch_batches = shuffled_batches(len(y), batch_size, torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    X, y = X.to(device), y.to(device)
    for _ in range(epochs):
        batches = epoch_batches()
        for idx in torch.cat(batches).to(device).split([len(b) for b in batches]):   # one host-to-device copy per epoch
            optimizer.zero_grad()
            F.cross_entropy(probe(X[idx]), y[idx]).backward()
            optimizer.step()
    return probe.eval()


@torch.no_grad()
def accuracy(probe, X, y):
    device = probe.weight.device
    return float((probe(X.to(device)).argmax(-1).cpu() == y).double().mean())


def train_regression(X, y, seed, device, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE):
    """Returns the trained nn.Linear(d, 1) (on `device`). X float32 [n, d], y float32 [n] (centred targets).
    As in the paper's n-gram runs, initialisation and shuffling draw from the global torch generator seeded here."""
    torch.manual_seed(seed)
    probe = torch.nn.Linear(X.shape[1], 1).to(device)
    loader = DataLoader(_Indices(len(y)), batch_size=batch_size, shuffle=True, collate_fn=lambda b: b)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    X, y = X.to(device), y.to(device)
    for _ in range(epochs):
        batches = list(loader)
        for idx in torch.cat(batches).to(device).split([len(b) for b in batches]):
            optimizer.zero_grad()
            F.mse_loss(probe(X[idx]).squeeze(-1), y[idx]).backward()
            optimizer.step()
        next(iter(loader))    # the paper runs drew one more shuffle per epoch (a pass for training metrics)
    return probe.eval()


@torch.no_grad()
def r2(probe, X, y):
    """Held-out R^2 = 1 - MSE / variance of y (baseline: the mean of the evaluated targets), in float32."""
    device = probe.weight.device
    y = y.to(device)
    mse = (probe(X.to(device)).squeeze(-1) - y).pow(2).mean()
    return float(1 - mse / (y.var(unbiased=False) + 1e-12))
