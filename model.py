import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

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


        # Define english and french embedding layers:
        print("model init",self.window_size, self.embedding_size, self.vocab_size)
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        # Create positional encoder layers
        self.pos_encode = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=False)

        # Define dense layer(s)
        # TODO: may not be supposed to use softmax on last one
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(2, activation="softmax")

    @tf.function
    def call(self, input):
        """
        :param input: batched ids corresponding to french sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # 1) Add the positional embeddings to french sentence embeddings
        embedding = self.embedding_layer(input)

        pos_embedding = self.pos_encode.call(embedding)

        # 2) Pass the french sentence embeddings to the encoder
        encoded = self.encoder.call(pos_embedding)

        flat = self.flatten(encoded)

        # 3) Apply dense layer(s) to the decoder out to generate probabilities
        out = self.dense_layer(flat)

        return out

    def accuracy_function(self, probabilities, labels, vocab, batch_inputs):
        """
        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """
        argmaxProbabilities = np.argmax(probabilities, axis=1)

        # code to print out highly polarized tweets!
        for i in range(len(probabilities)):
            index = argmaxProbabilities[i]
            prob = probabilities[i][index]
            if (prob > .95):
                tweet = ""
                for val in batch_inputs[i]:
                    word = list(vocab.keys())[val]
                    tweet += word + " "
                print("Highly polarized tweet: ", tweet)
                print("Prediction: ", index)
                print("Correct label: ", labels[i])

        comparison = (argmaxProbabilities == labels)
        return np.mean(comparison)

    def loss_function(self, prbs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass

        :param logits:  float tensor, word prediction probabilities [batch_size x 2]
        :param labels:  integer tensor, word prediction labels [batch_size]
        :return: the loss of the model as a tensor
        """
        dem_loss = 0
        rep_loss = 0
        for i in range(len(prbs)):
            if labels[i] == 1:
                dem_loss += -tf.math.log(prbs[i][1])
            if labels[i] == 0:
                rep_loss += -tf.math.log(prbs[i][0])
        return (dem_loss + rep_loss) / self.batch_size
