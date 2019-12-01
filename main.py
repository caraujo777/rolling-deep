import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import Model
import sys


def train(model, train, eng_padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train: train data (all data for training) of shape (num_sentences, 14)
    :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """
    at = 0
    while at + model.batch_size < len(train):
        batch = train[at: at + model.batch_size]
        at += model.batch_size

        labels = batch[:, :-1]
        mask = [el != eng_padding_index for el in labels]

        with tf.GradientTape() as tape:
            logits = model.call(batch)
            loss = model.loss_function(logits, labels, mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return


def test(model, test, padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initilized model to use for forward and backward pass
    :param test:  test data (all data for testing)
    :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :returns: perplexity of the test set, per symbol accuracy on test set
    """
    num_batches = 0
    total_loss = 0
    correct_words = 0
    non_padding_words = 0
    at = 0
    while at + model.batch_size < len(test):
        batch = test[at: at + model.batch_size]
        at += model.batch_size

        labels = batch[:, :-1]
        mask = [el != padding_index for el in labels]
        # edit: training
        logits = model.call(batch)
        loss = model.loss_function(logits, labels, mask)

        non_padding_word = np.sum(mask)
        acc = model.accuracy_function(logits, labels, mask) * non_padding_word
        non_padding_words += non_padding_word
        correct_words += acc

        total_loss += loss
        num_batches += 1
    avg_loss = total_loss / num_batches
    perplexity = tf.exp(avg_loss)
    return tf.reduce_mean(perplexity), correct_words / non_padding_words


def main():
    print("Running preprocessing...")
    # TODO: get data from parsed_climate.txt into right list format
    inputs, labels, vocab, padding_index = get_data('parsed_climate_inputs.txt', 'parsed_climate_labels.txt')
    print("Preprocessing complete.")

    #split into training and testing data!
    percentage_training = 0.8
    size_training = int(np.floor(percentage_training * len(inputs)))
    size_testing = len(inputs) - size_training
    training_data_inputs = inputs[:size_training]
    training_data_labels = labels[:size_training]
    test_data_inputs = inputs[size_training:]
    test_data_labels = labels[size_training:]


    model_args = (WINDOW_SIZE, len(vocab))
    model = Model(*model_args)

    print("Model Initialized!")
    # Train and Test Model for 1 epoch.
    train(model, training_data_inputs, padding_index)


    perplexity, acc = test(model, test, padding_index)
    # Print out perplexity
    print("per", perplexity)
    print("acc", acc)


if __name__ == '__main__':
    main()
