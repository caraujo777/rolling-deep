import numpy as np
import tensorflow as tf
import numpy as np

WINDOW_SIZE = 45 # max characters in a tweet: 280; avg number of characters per word: 6; 280/6 ~= 45

def get_data(inputs, labels):
     # Read data!
    x_data = []
    y_data = []
    with open(inputs, 'rt', encoding='latin') as data_file:
        for line in data_file: x_data.append(line)
    with open(labels, encoding='latin') as data_file:
        for line in data_file: y_data.append(int(line[0]))

    vocab = set((" ".join(x_data)).split())
    word2id = {w: i for i, w in enumerate(list(vocab))}
    new_x_data = []
    for line in x_data:
        newLine = []
        for word in line.split():
            newLine.append(word2id[word])
        new_x_data.append(newLine)
    x_data = new_x_data

    # pad end
    x_data = tf.keras.preprocessing.sequence.pad_sequences(x_data, padding='post')

    return x_data, y_data, word2id
