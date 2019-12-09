import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import Model
import sys


def train(model, train_inputs, train_labels, padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: the train inputs of shape (?)
    :param train_labels: the labels of the politcal party for train inputs
    :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """
    i = 0
    while (i < len(train_labels)):
        start = i
        i += model.batch_size
        end = i
        # for when it is not divisible by batch_size to not go out of bounds
        if i >= len(train_labels):
            break

        # batch sized inputs and labels!
        batch_inputs = train_inputs[start:end] # batch of tweets
        batch_labels = train_labels[start:end] # batch of labels for the tweets

        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs)
            loss = model.loss_function(logits, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels, padding_index, vocab):
    """
    Runs through one epoch - all testing examples.

    :param model: the initilized model to use for forward and backward pass
    :param test_inputs:  test data inputs (ie tweets)
    :param test_labels:  test data labels (ie political party corresponding to tweets)
    :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :returns: accuracy of test set
    """
    i = 0
    total_accuracy = 0
    num_batches = 0
    overall_list_tweets = []
    while (i < len(test_inputs)): # can use either french or english - same size!
        start = i
        i += model.batch_size
        end = i
        # for when it is not divisible by batch_size to chop it off
        if i >= len(test_inputs):
            break
        num_batches += 1
        # split up by inputs and labels
        batch_inputs = test_inputs[start:end] # batch of tweets
        batch_labels = test_labels[start:end] # batch of labels for the tweets
        probabilities = model.call(batch_inputs)
        batch_accuracy, list_tweets = model.accuracy_function(probabilities, batch_labels, vocab, batch_inputs)
        overall_list_tweets = overall_list_tweets + list_tweets
        total_accuracy += batch_accuracy
    return (total_accuracy / num_batches)

def main():
    print("Running preprocessing...")
    # TODO: get data from parsed_climate.txt into right list format
    inputs, labels, vocab, padding_index = get_data('parsed_climate_inputs.txt', 'parsed_climate_labels.txt')
    print("Preprocessing complete.")

    #split into training and testing data!
    percentage_training = 0.8
    size_training = int(np.floor(percentage_training * len(inputs)))

    training_data_inputs = inputs[:size_training]
    training_data_labels = labels[:size_training]
    test_data_inputs = inputs[size_training:]
    test_data_labels = labels[size_training:]

    model_args = (len(vocab), WINDOW_SIZE)
    model = Model(*model_args)

    print("Model Initialized!")
    # Train and Test Model for 1 epoch.
    num_epochs = 10
    for i in range(num_epochs):
        train(model, training_data_inputs, training_data_labels, padding_index)

        acc = test(model, test_data_inputs, test_data_labels, padding_index, vocab)
        # Print out perplexity
        print("at epoch", i, "acc", acc)



if __name__ == '__main__':
    main()
