import numpy as np
import tensorflow as tf
import numpy as np


PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 45 #max characters in a tweet: 280; avg number of characters per word: 6; 280/6 ~= 45


def get_data(inputs, labels):
    """
    Use the helper functions in this file to read and parse training and test data, then pad the corpus.
    Then vectorize your train and test data based on your vocabulary dictionaries.

    :param inputs: Path to the file with tweets
    :param labels: Path to the file with labels

    :return: Tuple of train containing:
    (2-d list or array with training sentences in vectorized/id form [num_sentences x 15] ),
    (2-d list or array with test sentences in vectorized/id form [num_sentences x 15]),
    vocab (Dict containg word->index mapping),
    english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
    """

     # 1) Read data!
    x_data = []
    y_data = []
    with open(inputs, 'rt', encoding='latin') as data_file:
        for line in data_file: x_data.append(line)
    with open(labels, encoding='latin') as data_file:
        for line in data_file: y_data.append(int(line[0]))

    # Build Vocabulary (word id's) from titles
    vocab = set((" ".join(x_data)).split())  # {'the', 'garden', 'hallway', 'to'..}
    word2id = {w: i for i, w in enumerate(list(vocab))}
    new_x_data = []
    for line in x_data:
        newLine = []
        for word in line.split():
            newLine.append(word2id[word])
        new_x_data.append(newLine)
    x_data = new_x_data

    # pad
    x_data = tf.keras.preprocessing.sequence.pad_sequences(x_data, padding='post')

    return x_data, y_data, vocab
