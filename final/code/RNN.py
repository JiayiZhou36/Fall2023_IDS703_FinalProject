import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time


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


def train_rnn_model(file_path_1, file_path_2):
    # Read data and create vocabulary
    music_data = read_file_to_sentences(file_path_1)
    sports_data = read_file_to_sentences(file_path_2)
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
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=pad_sequences
    )

    print("Start training")
    # Training loop
    num_epochs = 20
    start_time = time.time()  # Record the start time for training
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Record the start time for the epoch
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels.squeeze()).sum().item()
            total_samples += labels.size(0)

        epoch_end_time = time.time()  # Record the end time for the epoch
        epoch_time = (
            epoch_end_time - epoch_start_time
        )  # Calculate the time taken for the epoch

        # Calculate accuracy and average loss for the epoch
        accuracy = correct_predictions / total_samples
        average_loss = total_loss / len(dataloader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f} seconds"
        )

    total_training_time = time.time() - start_time  # Calculate the total training time
    print(f"Total training time: {total_training_time:.2f} seconds")


if __name__ == "__main__":
    # Calling function
    train_rnn_model("../data/category10.txt", "../data/category17.txt")
    train_rnn_model("../data/synthetic_music.txt", "../data/synthetic_sports.txt")
