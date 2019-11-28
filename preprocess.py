import numpy as np
import tensorflow as tf
import numpy as np


PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 14


def pad_corpus(text):
    """
    arguments are lists of sentences. Returns sents. The
    text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
    the end.

    :param text: list of sentences
    :return: list of padded sentences
    """
    padded_sentences = []
    sentence_lengths = []
    for line in text:
        padded = line[:WINDOW_SIZE]
        padded += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded) - 1)
        padded_sentences.append(padded)

    return padded_sentences


def build_vocab(sentences):
    """
    Builds vocab from list of sentences

    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN, PAD_TOKEN, UNK_TOKEN] + tokens)))

    vocab = {word: i for i, word in enumerate(all_words)}

    return vocab, vocab[PAD_TOKEN]


def convert_to_id(vocab, sentences):
    """
    Convert sentences to indexed

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
    return np.stack(
        [[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
    """
    Load text data from file

    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
  """
    text = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        for line in data_file: text.append(line.split())
    return text


def get_data(training_file, test_file):
    """
    Use the helper functions in this file to read and parse training and test data, then pad the corpus.
    Then vectorize your train and test data based on your vocabulary dictionaries.

    :param training_file: Path to the english training file.
    :param test_file: Path to the french test file.

    :return: Tuple of train containing:
    (2-d list or array with training sentences in vectorized/id form [num_sentences x 15] ),
    (2-d list or array with test sentences in vectorized/id form [num_sentences x 15]),
    vocab (Dict containg word->index mapping),
    english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
    """

    # 1) Read English and French Data for training and testing (see read_data)
    training_data = read_data(training_file)
    test_data = read_data(test_file)

    # 2) Pad training data (see pad_corpus)
    training_padded = pad_corpus(training_data)

    # 3) Pad testing data (see pad_corpus)
    test_padded = pad_corpus(test_data)

    # 4) Build vocab for french (see build_vocab)
    vocab, padding_index = build_vocab(training_padded)

    # 6) Convert training and testing english sentences to list of IDS (see convert_to_id)
    training_ids = convert_to_id(vocab, training_padded)
    test_ids = convert_to_id(vocab, test_padded)

    # train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index
    return training_ids, test_ids, vocab, padding_index

