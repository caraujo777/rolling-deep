import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import Model
import sys

"""
Runs through one epoch - all training examples.

:param model: the initilized model
:param test_inputs:  train data inputs (ie tweets)
:param test_labels:  train data labels (ie political party corresponding to tweets
"""
def train(model, train_inputs, train_labels):
    indices = range(len(train_inputs))
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)

    i = 0
    while (i < len(shuffled_inputs)):
        start = i
        i += model.batch_size
        end = i
        # for when it is not divisible by batch_size to not go out of bounds
        if i >= len(shuffled_labels):
            break

        # batch sized inputs and labels!
        batch_inputs = shuffled_inputs[start:end] # batch of tweets
        batch_labels = shuffled_labels[start:end] # batch of labels for the tweets

        with tf.GradientTape() as tape:
            logits = model.call(batch_inputs)
            loss = model.loss_function(logits, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples.

    :param model: the initilized model to use for forward and backward pass
    :param test_inputs:  test data inputs (ie tweets)
    :param test_labels:  test data labels (ie political party corresponding to tweets)
    :returns: accuracy of test set
    """
    i = 0
    total_accuracy = 0
    num_batches = 0
    while (i < len(test_inputs)):
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
        batch_accuracy = model.accuracy_function(probabilities, batch_labels)
        total_accuracy += batch_accuracy
    return total_accuracy / num_batches

def main():
    print("Running preprocessing...")
    inputs, labels, vocab = get_data('parsed_climate_inputs.txt', 'parsed_climate_labels.txt')
    print("Preprocessing complete.")

    #split into training and testing data!
    percentage_training = 0.8
    size_training = int(np.floor(percentage_training * len(inputs)))

    model_args = (len(vocab), WINDOW_SIZE)
    model = Model(*model_args)

    print("Model Initialized!")
    # Train and Test Model for 1 epoch.
    num_epochs = 20

    indices = range(len(inputs))
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(inputs, shuffled_indices)
    shuffled_labels = tf.gather(labels, shuffled_indices)


    for i in range(num_epochs):
        training_data_inputs = shuffled_inputs[:size_training]
        training_data_labels = shuffled_labels[:size_training]
        test_data_inputs = shuffled_inputs[size_training:]
        test_data_labels = shuffled_labels[size_training:]

        train(model, training_data_inputs, training_data_labels)


        test_acc = test(model, test_data_inputs, test_data_labels)
        train_acc = test(model, training_data_inputs, training_data_labels)
        # Print out perplexity
        print("at epoch", i, "test acc ", test_acc, "train acc", train_acc)


if __name__ == '__main__':
    main()
