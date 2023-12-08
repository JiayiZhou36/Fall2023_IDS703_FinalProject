import random
from typing import List, Mapping, Optional, Sequence, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import time

FloatArray = NDArray[np.float64]


def read_file_to_sentences(file_path: str) -> List[List[str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip().split(",") for line in file if line.strip()]


def create_vocabulary_and_mapping(
    music: List[List[str]], sports: List[List[str]]
) -> Tuple[List[Optional[str]], Dict[Optional[str], int]]:
    vocabulary = sorted(
        set(token for sentence in music + sports for token in sentence)
    ) + [None]
    vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}
    return vocabulary, vocabulary_map


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def sum_token_embeddings(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    return np.array(token_embeddings).sum(axis=0)


def split_train_test(
    X: FloatArray, y: FloatArray, test_percent: float = 10
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    N = len(y)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]
    testing_idx = data_idx[:break_idx]
    return X[training_idx, :], y[training_idx], X[testing_idx, :], y[testing_idx]


def generate_data_token_counts(
    music_document: List[List[str]],
    sports_document: List[List[str]],
    vocabulary_map: Mapping[Optional[str], int],
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    X: FloatArray = np.array(
        [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in music_document + sports_document
        ]
    )
    y: FloatArray = np.array([0] * len(music_document) + [1] * len(sports_document))
    return split_train_test(X, y)


def generate_data_tfidf(
    X_train: FloatArray, X_test: FloatArray
) -> Tuple[FloatArray, FloatArray]:
    tfidf = TfidfTransformer(norm=None).fit(X_train)
    return tfidf.transform(X_train), tfidf.transform(X_test)


def run_experiment(file_path_1, file_path_2) -> None:
    random.seed(31)
    music = read_file_to_sentences(file_path_1)
    sports = read_file_to_sentences(file_path_2)
    vocabulary, vocabulary_map = create_vocabulary_and_mapping(music, sports)

    start_time = time.time()
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        music, sports, vocabulary_map
    )
    clf = MultinomialNB().fit(X_train, y_train)
    print("raw counts (train):", clf.score(X_train, y_train))
    print("raw_counts (test):", clf.score(X_test, y_test))
    elapsed_time = time.time() - start_time
    print(f"Time for raw counts section: {elapsed_time:.2f} seconds")

    start_time = time.time()
    X_train_tfidf, X_test_tfidf = generate_data_tfidf(X_train, X_test)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    print("tfidf (train):", clf.score(X_train_tfidf, y_train))
    print("tfidf (test):", clf.score(X_test_tfidf, y_test))
    elapsed_time = time.time() - start_time
    print(f"Time for tfidf section: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    run_experiment("../data/category10.txt", "../data/category17.txt")
    run_experiment("../data/synthetic_music.txt", "../data/synthetic_sports.txt")
