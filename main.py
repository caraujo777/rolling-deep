import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys


def train(model, train, eng_padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """

    # NOTE: For each training step, you should pass in the french sentences to be used by the encoder,
    # and english sentences to be used by the decoder
    # - The english sentences passed to the decoder have the last token in the window removed:
    #	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
    #
    # - When computing loss, the decoder labels should have the first word removed:
    #	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]

    at = 0
    while at + model.batch_size < len(train):
        batch = train[at: at + model.batch_size]
        at += model.batch_size

        labels = batch[:, :-1]
        mask = [el != eng_padding_index for el in labels]

        # edit: training
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
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :returns: perplexity of the test set, per symbol accuracy on test set
    """
    # Note: Follow the same procedure as in train() to construct batches of data!
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
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN", "TRANSFORMER"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [RNN/TRANSFORMER]")
        exit()

    print("Running preprocessing...")
    train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data(
        'data/fls.txt', 'data/els.txt', 'data/flt.txt', 'data/elt.txt')
    print("Preprocessing complete.")

    model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))
    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)
    elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args)

    # Train and Test Model for 1 epoch.
    train(model, train_french, train_english, eng_padding_index)
    perplexity, acc = test(model, test_french, test_english, eng_padding_index)
    # Print out perplexity
    print("per", perplexity)
    print("acc", acc)


if __name__ == '__main__':
    main()


