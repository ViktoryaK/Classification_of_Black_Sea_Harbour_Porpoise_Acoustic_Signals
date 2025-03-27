import torch
import torch.nn as nn
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar, z


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=64):
        super(SiameseNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=embedding_dim, batch_first=True)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward_one(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def forward(self, input1, input2):
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        return out1, out2


# Contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


def train_siamese(model, train_loader, criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_pairs1, batch_pairs2, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_pairs1, batch_pairs2, batch_labels = batch_pairs1.to(device), batch_pairs2.to(
                device), batch_labels.to(device)

            optimizer.zero_grad()

            output1, output2 = model(batch_pairs1, batch_pairs2)
            loss = criterion(output1, output2, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    print("Training complete.")


def evaluate_siamese(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_pairs1, batch_pairs2, batch_labels in tqdm(val_loader, desc="Evaluating"):
            batch_pairs1, batch_pairs2, batch_labels = batch_pairs1.to(device), batch_pairs2.to(
                device), batch_labels.to(device)

            output1, output2 = model(batch_pairs1, batch_pairs2)
            euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance < 1.0).float()

            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / total * 100
    print(f"Model Accuracy: {accuracy:.2f}%")
