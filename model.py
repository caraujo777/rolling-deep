import numpy as np
import tensorflow as tf


# TODO: get sentence embeddings, maybe used SVD or something
# TODO: train model (transformer, or other) to predict label

class Model(tf.keras.Model):
    def __init__(self, vocab_size, window_size):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size

        # 1) Define any hyperparameters
        # 2) Define embeddings, encoder, decoder, and feed forward layers

        # Define batch size and optimizer/learning rate
        self.batch_size = 150
        self.embedding_size = 32

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # TODO: idk if we actually need all these layers, this code is modified off homework 4

        # Define english and french embedding layers:
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size,
                                                                input_length=self.window_size)
        # Create positional encoder layers
        # self.pos_encode = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
        #
        # # Define encoder and decoder layers:
        # self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=False)
        #
        # # Define dense layer(s)
        # self.dense_layer = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    @tf.function
    def call(self, input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # 1) Add the positional embeddings to french sentence embeddings
        embedding = self.embedding_layer(input)
        pos_embedding = self.pos_encode.call(embedding)

        # 2) Pass the french sentence embeddings to the encoder
        encoded = self.encoder.call(pos_embedding)

        # 3) Apply dense layer(s) to the decoder out to generate probabilities
        out = self.dense_layer(encoded)

        return out

    def accuracy_function(self, prbs, labels, mask):
        """
        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
        masked = tf.boolean_mask(loss, mask)
        return tf.reduce_mean(masked)
