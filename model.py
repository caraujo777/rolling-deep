import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Model(tf.keras.Model):
    def __init__(self, vocab_size, window_size):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size

        # Define batch size and optimizer/learning rate
        self.batch_size = 32
        self.embedding_size = 128
        self.lstm_output = 200

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, mask_zero=True))
        self.model.add(tf.keras.layers.LSTM(self.lstm_output, dropout=0.2))
        self.model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    @tf.function
    def call(self, input):
        """
        :param input: batched ids corresponding to french sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        return self.model(input)

    def accuracy_function(self, probabilities, labels, vocab, batch_inputs):
        """
        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """
        argmaxProbabilities = np.argmax(probabilities, axis=1)
        comparison = (argmaxProbabilities == labels)
        return np.mean(comparison), list_tweets

    def loss_function(self, prbs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass

        :param logits:  float tensor, word prediction probabilities [batch_size x 2]
        :param labels:  integer tensor, word prediction labels [batch_size]
        :return: the loss of the model as a tensor
        """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=False)
