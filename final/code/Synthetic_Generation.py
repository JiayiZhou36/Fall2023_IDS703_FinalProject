import random
from collections import Counter
import nltk
from nltk.util import ngrams

# Ensure the necessary NLTK data is downloaded
nltk.download("punkt")


def read_and_process_file(file_path, n):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        print("Number of lines read:", len(lines))  # Debug print

        all_tags = [tag for line in lines for tag in line.strip().split(",")]
        print("Sample tags:", all_tags[:10])  # Debug print

        sentence_lengths = [len(line.strip().split(",")) for line in lines]
        print("Sentence lengths:", sentence_lengths[:10])  # Debug print

        n_grams = list(ngrams(all_tags, n))
        print("Sample n-grams:", n_grams[:10])  # Debug print

        return n_grams, sentence_lengths


def generate_synthetic_data(n_grams, sentence_lengths, n, max_rows=None):
    synthetic_data = []
    n_gram_counts = Counter(n_grams)
    total_rows = 0

    for length in sentence_lengths:
        sentence = []
        for i in range(length):
            if n > 1 and i < n - 1:
                # For n-grams (n > 1), randomly select a start sequence
                sentence.extend(random.choice(n_grams)[: n - 1])
            else:
                # For unigrams (n = 1) or generating subsequent tags in n-grams
                last_n_1 = tuple(sentence[-(n - 1) :]) if n > 1 else tuple()
                possible_next_tags = [
                    gram[-1] for gram in n_gram_counts if gram[:-1] == last_n_1
                ]
                if possible_next_tags:
                    next_tag = random.choice(possible_next_tags)
                    sentence.append(next_tag)
                else:
                    # If no possible continuation, use a random tag
                    sentence.append(random.choice(list(n_gram_counts.keys()))[0])

        synthetic_data.append(sentence[:length])  # Ensure correct sentence length
        print("Generated sentence:", ",".join(sentence))

        total_rows += 1
        if max_rows is not None and total_rows >= max_rows:
            break

    return synthetic_data


def write_synthetic_data(synthetic_data, output_file):
    """
    Writes the synthetic data to a file, ensuring each set of tags is joined by commas.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for sentence in synthetic_data:
            file.write(",".join(sentence) + "\n")


# Parameters
n = 1  # Unigram
max_rows = 10000  # Maximum number of rows

# Process files and generate synthetic data
for category, output_file in [
    ("../data/category10.txt", "synthetic_music.txt"),
    ("../data/category17.txt", "synthetic_sports.txt"),
]:
    n_grams, sentence_lengths = read_and_process_file(category, n)
    synthetic_data = generate_synthetic_data(n_grams, sentence_lengths, n, max_rows)
    print(synthetic_data)
    write_synthetic_data(synthetic_data, output_file)
