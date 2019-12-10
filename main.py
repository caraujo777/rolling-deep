import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from model import Model
import sys


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

def create_heatmap(inputs, labels, word2id):
    # swap keys and vals in dict
    id2word = dict([(value, key) for key, value in word2id.items()])

    with open("input_data/keywords.txt", "r") as f:
        for line in f:
            words = list(line.replace("\n","").split(', '))
    f.close()
    words_count_dem = {}
    words_count_rep = {}
    used_word = {}
    
    for word in words:
        used_word[word] = False

    list_tweets = []
    for i in range(len(inputs)):
        tweet=""
        for val in inputs[i]:
            print(val)
            word = id2word[val]
            tweet += word + " "
        list_tweets.append(tweet)
        if(i > 1000):
            break

    for i in range(len(list_tweets)):
        text = list_tweets[i]
        print(text)
        curr = words_count_dem if labels[i] == 1 else words_count_rep
        for word in words:
            cont = True
            text_words = text.split()
            for single_word in word.split():
                if single_word not in text_words and single_word+"s":
                    cont = False
                if cont and not used_word[word]:
                    used_word[word] = True
                    curr[word] = curr.get(word, 0) + 1
                    # print(word, curr[word])
                    # print('/n')
        for word in words:
            used_word[word]= False
    
    # print(words_count_dem)
    # print(words_count_rep)
        
        
def test(model, test_inputs, test_labels, vocab):
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
        batch_accuracy = model.accuracy_function(probabilities, batch_labels, vocab, batch_inputs)
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

    create_heatmap(shuffled_inputs, shuffled_labels, vocab)

    # for i in range(num_epochs):
    #     training_data_inputs = shuffled_inputs[:size_training]
    #     training_data_labels = shuffled_labels[:size_training]
    #     test_data_inputs = shuffled_inputs[size_training:]
    #     test_data_labels = shuffled_labels[size_training:]

    #     train(model, training_data_inputs, training_data_labels)


    #     test_acc = test(model, test_data_inputs, test_data_labels, vocab)
    #     train_acc = test(model, training_data_inputs, training_data_labels, vocab)
    #     # Print out perplexity
    #     print("at epoch", i, "test acc ", test_acc, "train acc", train_acc)


if __name__ == '__main__':
    main()
