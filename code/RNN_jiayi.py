import random
from typing import List, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def read_file_to_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().split(",") for line in file if line.strip()]


# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out


# Define a custom dataset with sequence padding
class CustomDataset(Dataset):
    def __init__(self, data, labels, vocab):
        self.data = data
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        indexed_data = [self.vocab[word] for word in self.data[idx]]
        return torch.LongTensor(indexed_data), torch.LongTensor([self.labels[idx]])


def pad_sequences(batch):
    # Separate input sequences and labels
    sequences, labels = zip(*batch)

    # Pad input sequences
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [
        torch.cat([seq, torch.zeros(max_len - len(seq)).long()]) for seq in sequences
    ]
    padded_sequences = torch.stack(padded_sequences)

    # Convert labels to a tensor
    labels = torch.LongTensor(labels)

    return padded_sequences, labels


music_data = read_file_to_sentences("../data/category10.txt")
sports_data = read_file_to_sentences("../data/category17.txt")
print("Read files done")

all_data = music_data + sports_data
labels = [0] * len(music_data) + [1] * len(sports_data)  # 0 for music, 1 for sports

# Flatten the list of lists
all_data_flat = [word for sublist in all_data for word in sublist]

# Create a vocabulary for embedding
vocab = {word: idx for idx, word in enumerate(set(all_data_flat))}

# Create the RNN model
input_size = len(vocab)
hidden_size = 64
output_size = 2  # Two categories: music and sports
model = SimpleRNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader for training with sequence padding
dataset = CustomDataset(all_data, labels, vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_sequences)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
