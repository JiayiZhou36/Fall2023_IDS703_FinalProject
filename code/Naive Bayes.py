# Import Packages
import pandas as pd

import random
from typing import List, Mapping, Optional, Sequence

import nltk
import numpy as np
from numpy.typing import NDArray


from unigram_model_numpy import UnigramModel

# Import Pretrained Data
import random
from typing import List, Mapping, Optional, Sequence
import gensim
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.float64]
import gensim.downloader as api

# Load Google's pre-trained Word2Vec model.

model = api.load("word2vec-google-news-300")
# print(api.info())  # show info about available models/datasets

# Un-comment this to fix the random seed
random.seed(31)


# Read the file and return a list of sentences
def read_file_to_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().split(",") for line in file if line.strip()]


music = read_file_to_sentences(
    "/Users/anniewu/Desktop/2023 Fall/IDS 703/Final Project/category10.txt"
)
sports = read_file_to_sentences(
    "/Users/anniewu/Desktop/2023 Fall/IDS 703/Final Project/category17.txt"
)
# gaming = read_file_to_sentences("category20.txt")
# entertainment = read_file_to_sentences("category24.txt")


# Created a vocaulary map for the tokens from the four categories

# Created a unique vocabulary list from the four categories
vocabulary = sorted(
    set(
        token
        for sentence in music + sports
        # + gaming + entertainment
        for token in sentence
    )
) + [None]

# Create a vocabulary map for the tokens from the four categories
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}


# Embedding
# One Hot Encoding
def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def encode_document(tokens: Sequence[Optional[str]]) -> List[FloatArray]:
    """Apply one-hot encoding to each document."""
    encodings = [onehot(vocabulary_map, token) for token in tokens]
    return encodings


# Assemble Training and Testing Dataset

# assemble training and testing data
music_observations = [(encode_document(sentence), 0) for sentence in music]
sports_observations = [(encode_document(sentence), 1) for sentence in sports]
# gaming_observations = [(encode_document(sentence), 2) for sentence in gaming]
# entertainment_observations = [
# (encode_document(sentence), 3) for sentence in entertainment
# ]

print(f"music observations: {music_observations}")

all_data = (
    music_observations
    + music_observations
    # + gaming_observations
    # + entertainment_observations
)

print(f"all data: {all_data}")

random.shuffle(all_data)
test_percent = 10
break_idx = round(test_percent / 100 * len(all_data))
training_data = all_data[break_idx:]
testing_data = all_data[:break_idx]


# Train the Model
# train Naive Bayes

# Create document for each category
# Each document has the tokens and the label
music_documents = [
    observation[0] for observation in training_data if observation[1] == 0
]
sports_documents = [
    observation[0] for observation in training_data if observation[1] == 1
]
# gaming_documents = [
# observation[0] for observation in training_data if observation[1] == 2
# ]
# entertainment_documents = [
# observation[0] for observation in training_data if observation[1] == 3
# ]

# Train Model for each category
music_language_model = UnigramModel(len(vocabulary_map))
music_language_model.train(
    [token for document in music_documents for token in document]
)

sports_language_model = UnigramModel(len(vocabulary_map))
sports_language_model.train(
    [token for document in sports_documents for token in document]
)

# gaming_language_model = UnigramModel(len(vocabulary_map))
# gaming_language_model.train(
# [token for document in gaming_documents for token in document]
# )

# entertainment_language_model = UnigramModel(len(vocabulary_map))
# entertainment_language_model.train(
# [token for document in entertainment_documents for token in document]
# )

# Calculate the prior probability for each category
pmusic = len(music_documents) / len(training_data)
psports = len(sports_documents) / len(training_data)
# pgaming = len(gaming_documents) / len(training_data)
# pentertainment = len(entertainment_documents) / len(training_data)

num_correct = 0
for document, label in testing_data:
    # apply model for each class
    music_logp_unnormalized = music_language_model.apply(document) + np.log(pmusic)
    sports_logp_unnormalized = sports_language_model.apply(document) + np.log(psports)
    # gaming_logp_unnormalized = gaming_language_model.apply(document) + np.log(pgaming)
    # entertainment_logp_unnormalized = entertainment_language_model.apply(
    # document
    # ) + np.log(pentertainment)

    # normalize
    logp_data = np.logaddexp(
        music_logp_unnormalized,
        sports_logp_unnormalized,
        # gaming_logp_unnormalized,
        # entertainment_logp_unnormalized,
    )
    music_logp = music_logp_unnormalized - logp_data
    sports_logp = sports_logp_unnormalized - logp_data
    # gaming_logp = gaming_logp_unnormalized - logp_data
    # entertainment_logp = entertainment_logp_unnormalized - logp_data

    # make guess
    pc0 = np.exp(music_logp)
    pc1 = np.exp(sports_logp)
    # pc2 = np.exp(gaming_logp)
    # pc3 = np.exp(entertainment_logp)
    guess = max(
        pc0,
        pc1
        # , pc2, pc3
    )
    print(
        pc0,
        pc1
        # , pc2, pc3
        ,
        guess,
        label,
    )

    if guess == label:
        num_correct += 1

print(num_correct / len(testing_data))
