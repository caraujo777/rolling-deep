import numpy as np
import tensorflow as tf
import numpy as np


def Self_Attention(K, V, Q, use_mask=False):
    """
    :param K: is [batch_size x window_size_keys x embedding_size]
    :param V: is [batch_size x window_size_values x embedding_size]
    :param Q: is [batch_size x window_size_queries x embedding_size]
    :return: attention
    """

    window_size_queries = Q.get_shape()[1]  # window size of queries
    window_size_keys = K.get_shape()[1]  # window size of keys
    mask = tf.convert_to_tensor(
        value=np.transpose(np.tril(np.ones((window_size_queries, window_size_keys)) * np.NINF, -1), (1, 0)),
        dtype=tf.float32)
    atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

    # TODO:
    mult = tf.linalg.matmul(Q, K, transpose_b=True)
    embedding_size = K.shape[2]
    scale = mult / np.sqrt(embedding_size)
    if use_mask:
        scale += atten_mask
    softmax = tf.nn.softmax(scale)
    return tf.linalg.matmul(softmax, V)


class Atten_Head(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, use_mask):
        super(Atten_Head, self).__init__()

        self.use_mask = use_mask
        self.WK = self.add_weight("WK", shape=(input_size, output_size))
        self.WV = self.add_weight("WV", shape=(input_size, output_size))
        self.WQ = self.add_weight("WQ", shape=(input_size, output_size))

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        :param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
        """

        K = tf.tensordot(inputs_for_keys, self.WK, axes=[[2], [0]])
        V = tf.tensordot(inputs_for_values, self.WV, axes=[[2], [0]])
        Q = tf.tensordot(inputs_for_queries, self.WQ, axes=[[2], [0]])

        return Self_Attention(K, V, Q, self.use_mask)


class Multi_Headed(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask):
        super(Multi_Headed, self).__init__()

class Feed_Forwards(tf.keras.layers.Layer):
    def __init__(self, emb_sz):
        super(Feed_Forwards, self).__init__()

        self.layer_1 = tf.keras.layers.Dense(emb_sz, activation='relu')
        self.layer_2 = tf.keras.layers.Dense(emb_sz)

    @tf.function
    def call(self, inputs):
        """
        :param inputs: input tensor [batch_size x window_size x embedding_size]
        :return: tensor [batch_size x window_size x embedding_size]
        """
        layer_1_out = self.layer_1(inputs)
        layer_2_out = self.layer_2(layer_1_out)
        return layer_2_out


class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_sz, is_decoder, multi_headed=False):
        super(Transformer_Block, self).__init__()

        self.ff_layer = Feed_Forwards(emb_sz)
        self.self_atten = Atten_Head(emb_sz, emb_sz, use_mask=is_decoder) if not multi_headed else Multi_Headed(emb_sz,
                                                                                                                use_mask=is_decoder)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.self_context_atten = Atten_Head(emb_sz, emb_sz, use_mask=False) if not multi_headed else Multi_Headed(
                emb_sz, use_mask=False)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def call(self, inputs, context=None):
        """
        :param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
        :context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
            default=None, This is context from the encoder to be used as Keys and Values in self-attention function
        """

        atten_out = self.self_atten(inputs, inputs, inputs)
        atten_out += inputs
        atten_normalized = self.layer_norm(atten_out)

        if self.is_decoder:
            assert context is not None, "Decoder blocks require context"

            context_atten_out = self.self_context_atten(atten_normalized, context, context)
            context_atten_out += atten_normalized
            atten_normalized = self.layer_norm(context_atten_out)

        ff_out = self.ff_layer(atten_normalized)
        ff_out += atten_normalized
        ff_norm = self.layer_norm(ff_out)

        return tf.nn.relu(ff_norm)


class Position_Encoding_Layer(tf.keras.layers.Layer):
    def __init__(self, window_sz, emb_sz):
        super(Position_Encoding_Layer, self).__init__()
        print(window_sz, emb_sz)
        self.positional_embeddings = self.add_weight("pos_embed", shape=[window_sz, emb_sz])

    @tf.function
    def call(self, x):
        """
        :param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
        :return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
        """
        print("hi",x.shape, self.positional_embeddings.shape)
        return x + self.positional_embeddings

