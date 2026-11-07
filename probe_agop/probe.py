"""Linear probe training (the recipe of the paper's classification probes).

Inputs are mean-centred with the float64 mean of the training representations (the same mean is subtracted from the
test set). The probe is a single nn.Linear layer trained with Adam (lr 1e-3, weight decay 1e-4 on weights and bias),
cross-entropy, batch 256, 10 epochs; the final epoch is kept. The probe seed fixes both the initialisation and the
shuffling order.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

LR, WEIGHT_DECAY, BATCH_SIZE, EPOCHS = 1e-3, 1e-4, 256, 10


def centre(train, test):
    mean = train.double().mean(0)
    return (train.double() - mean).float(), (test.double() - mean).float()


def train_probe(X, y, n_classes, seed, device, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE):
    """Returns the trained nn.Linear (on `device`). X float32 [n, d], y int64 [n]."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        probe = torch.nn.Linear(X.shape[1], n_classes)
    probe.to(device)
    shuffle = torch.Generator().manual_seed(seed)
    batches = DataLoader(range(len(y)), batch_size=batch_size, shuffle=True, generator=shuffle)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    X, y = X.to(device), y.to(device)
    for _ in range(epochs):
        for idx in batches:
            idx = idx.to(device)
            optimizer.zero_grad()
            F.cross_entropy(probe(X[idx]), y[idx]).backward()
            optimizer.step()
    return probe.eval()


@torch.no_grad()
def accuracy(probe, X, y):
    device = probe.weight.device
    return float((probe(X.to(device)).argmax(-1).cpu() == y).double().mean())
