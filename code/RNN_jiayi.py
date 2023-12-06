import random
from typing import List, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

# define type
FloatArray = NDArray[np.float64]


class RNN(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        MAX_VOCAB_SIZE = len(vocabulary_map)  # Update this line
        self.embedding = nn.Embedding(MAX_VOCAB_SIZE, self.embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, document: Sequence[torch.Tensor]) -> torch.Tensor:
        embedded = self.embedding(document)
        output, _ = self.rnn(embedded)
        output = self.fc(output[-1, :, :])
        return output


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    if len(vocabulary_map) < 2:
        raise ValueError("Vocabulary size must be at least 2 for one-hot encoding.")

    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding.reshape(1, -1)


def prepare_data(
    documents: List[List[str]], vocabulary_map: Mapping[str, int]
) -> Sequence[torch.Tensor]:
    data = []
    for sentence_list in documents:
        sentence_data = []
        for token in sentence_list:
            if token in vocabulary_map:
                sentence_data.append(vocabulary_map[token])
        data.append(torch.tensor(sentence_data, dtype=torch.long))
    return data


def read_file_to_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().split(",") for line in file if line.strip()]


# Read and process files
music = read_file_to_sentences("../data/category10.txt")
sports = read_file_to_sentences("../data/category17.txt")
print("Read files done")
vocabulary = sorted(set(token for sentence in music + sports for token in sentence)) + [
    None
]
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}
print("Vocabulary done")
MAX_VOCAB_SIZE = len(vocabulary_map)  # Define your desired maximum vocabulary size
music_data = prepare_data(music, vocabulary_map)
sports_data = prepare_data(sports, vocabulary_map)
print("Prepared data done")

# Create labels
y_true = [torch.tensor([0]) for _ in music_data] + [
    torch.tensor([1]) for _ in sports_data
]

# Combine datasets
X = music_data + sports_data

# Define RNN model
embedding_size = len(vocabulary_map)
hidden_size = 128
output_size = 2

model = RNN(embedding_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()


# Usage
def train_rnn_model(model, X, y_true, optimizer, loss_fn, num_epochs=100):
    total_loss = 0
    for epoch in range(num_epochs):
        for x_i, y_i in zip(X, y_true):
            optimizer.zero_grad()
            y_pred = model(x_i.unsqueeze(0))  # Adjust the input format
            loss = loss_fn(y_pred, y_i.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Final total loss:", total_loss)


train_rnn_model(model, X, y_true, optimizer, loss_fn, num_epochs=100)
