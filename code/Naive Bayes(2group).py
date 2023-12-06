import random
from typing import List, Mapping, Optional, Sequence
import gensim
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

FloatArray = NDArray[np.float64]
import gensim.downloader as api

# Load Google's pre-trained Word2Vec model.
model = api.load("word2vec-google-news-300")

# Un-comment this to fix the random seed
random.seed(31)


def read_file_to_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().split(",") for line in file if line.strip()]


music = read_file_to_sentences("category10.txt")
sports = read_file_to_sentences("category17.txt")

vocabulary = sorted(set(token for sentence in music + sports for token in sentence)) + [
    None
]

vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def sum_token_embeddings(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    """Sum the token embeddings."""
    total: FloatArray = np.array(token_embeddings).sum(axis=0)
    return total


def split_train_test(
    X: FloatArray, y: FloatArray, test_percent: float = 10
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split data into training and testing sets."""
    N = len(y)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]
    testing_idx = data_idx[:break_idx]
    X_train = X[training_idx, :]
    y_train = y[training_idx]
    X_test = X[testing_idx, :]
    y_test = y[testing_idx]
    return X_train, y_train, X_test, y_test


def generate_data_token_counts(
    music_document: list[list[str]],
    sports_document: list[list[str]],
) -> tuple[FloatArray, FloatArray]:
    """Generate training and testing data with raw token counts for the two categories."""

    # Aggregate embeddings for each category
    X: FloatArray = np.array(
        [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in music_document
        ]
        + [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in sports_document
        ]
    )
    # Generate labels for each category
    # Assuming music:0, sports:1
    y: FloatArray = np.array(
        [0 for sentence in music_document] + [1 for sentence in sports_document]
    )

    return split_train_test(X, y)


def generate_data_tfidf(
    music_document: list[list[str]],
    sports_document: list[list[str]],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with TF-IDF scaling."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        music_document, sports_document
    )
    tfidf = TfidfTransformer(norm=None).fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    return X_train, y_train, X_test, y_test


def generate_data_lsa(
    music_document: list[list[str]],
    sports_document: list[list[str]],
) -> tuple[FloatArray, FloatArray]:
    """Generate training and testing data with LSA."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        music_document, sports_document
    )
    lsa = TruncatedSVD(n_components=300).fit(X_train)
    X_train = lsa.transform(X_train)
    X_test = lsa.transform(X_test)
    return X_train, y_train, X_test, y_test


def generate_data_word2vec(
    music_document: list[list[str]],
    sports_document: list[list[str]],
) -> tuple[FloatArray, FloatArray]:
    """Generate training and testing data with word2vec."""
    # Load pretrained word2vec model from gensim
    model = api.load("word2vec-google-news-300")

    def get_document_vector(sentence: list[str]) -> NDArray:
        """Return document vector by summing word vectors."""
        vectors = [model[word] for word in sentence if word in model.key_to_index]
        if vectors:
            return np.sum(vectors, axis=0)
        else:
            return np.zeros(
                300
            )  # return zero vector if no word in the document has a pretrained vector

    # Produce document vectors for each sentence
    X = np.array(
        [get_document_vector(sentence) for sentence in music_document + sports_document]
    )
    y = np.array(
        [0 for sentence in music_document] + [1 for sentence in sports_document]
    )
    return split_train_test(X, y)


def run_experiment() -> None:
    """Compare performance with different embeddings using Naive Bayes."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(music, sports)
    clf = MultinomialNB().fit(X_train, y_train)
    print("raw counts (train):", clf.score(X_train, y_train))
    print("raw_counts (test):", clf.score(X_test, y_test))

    X_train, y_train, X_test, y_test = generate_data_tfidf(music, sports)
    clf = MultinomialNB().fit(X_train, y_train)
    print("tfidf (train):", clf.score(X_train, y_train))
    print("tfidf (test):", clf.score(X_test, y_test))


if __name__ == "__main__":
    run_experiment()
