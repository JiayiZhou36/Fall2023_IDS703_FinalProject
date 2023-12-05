import random
from typing import List, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

FloatArray = NDArray[np.float64]


class RNN(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1 = nn.Sequential(
            nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.l2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, document: Sequence[torch.Tensor]) -> torch.Tensor:
        hidden = torch.zeros((self.hidden_size, 1), requires_grad=True)
        for token_embedding in document:
            hidden = self.forward_cell(token_embedding, hidden)
        output = self.l2(hidden.T).T
        return output

    def forward_cell(
        self, token_embedding: torch.Tensor, previous_hidden: torch.Tensor
    ) -> torch.Tensor:
        concatenated = torch.cat((token_embedding, previous_hidden), dim=0)
        hidden = self.l1(concatenated.T).T
        return hidden


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    embedding = np.zeros((len(vocabulary_map), 1))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx, 0] = 1
    return embedding


def prepare_data(
    documents: List[List[str]], vocabulary_map: Mapping[str, int]
) -> Sequence[Sequence[torch.Tensor]]:
    return [
        [
            torch.tensor(onehot(vocabulary_map, token).astype("float32"))
            for token in sentence
        ]
        for sentence in documents
    ]


def read_file_to_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().split(",") for line in file if line.strip()]


# Read and process files
music = read_file_to_sentences("category10.txt")
sports = read_file_to_sentences("category17.txt")
vocabulary = sorted(set(token for sentence in music + sports for token in sentence)) + [
    None
]
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}

music_data = prepare_data(music, vocabulary_map)
sports_data = prepare_data(sports, vocabulary_map)

# Create labels
y_true = [torch.tensor([0]) for _ in music_data] + [
    torch.tensor([1]) for _ in sports_data
]

# Combine datasets
X = music_data + sports_data


# Define RNN model
def train_rnn_model(model, X, y_true, optimizer, loss_fn, num_epochs=100):
    total_loss = 0
    for epoch in range(num_epochs):
        for x_i, y_i in zip(X, y_true):
            optimizer.zero_grad()
            y_pred = model(x_i)
            loss = loss_fn(y_pred, y_i)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Final parameters:", list(model.parameters()))
    print("Final total loss:", total_loss)


# Usage
embedding_size = len(vocabulary_map)
hidden_size = 128
output_size = 2

model = RNN(embedding_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    train_rnn_model(model, X, y_true, optimizer, loss_fn, num_epochs=100)
