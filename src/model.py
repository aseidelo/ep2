import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import data_prep
import numpy as np

class TokenAndEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, tokenizer, embedding_file):
        super(TokenAndEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer
        pretrained_embed_matrix = self.set_embedding_matrix(embedding_file)
        self.embedder = layers.Embedding(input_dim=vocab_size+1, output_dim=embed_dim, weights=[pretrained_embed_matrix])

    def call(self, x): # ja tokenizado
        return self.embedder(x)

    def set_embedding_matrix(self, embedding_file):
        voc = self.tokenizer.get_vocabulary()
        unicode_voc = [s.decode() for s in voc]
        word_index = dict(zip(unicode_voc, range(len(unicode_voc))))
        embeddings_index = {}
        with open(embedding_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        print("Found %s word vectors." % len(embeddings_index))
        # Prepare embedding matrix
        hits = 0
        misses = 0
        embedding_matrix = np.zeros((self.vocab_size+1, self.embed_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits = hits + 1
            else:
                misses = misses + 1
        print("Converted %d words (%d misses)" % (hits, misses))
        return embedding_matrix

class LSTMEncoder(layers.Layer):
    def __init__(self, units, bidir=False):
        super(LSTMEncoder, self).__init__()
        self.lstm1 = layers.LSTM(units)
        #self.lstm2 = layers.LSTM(units)
        if (bidir):
            self.lstm1 = layers.Bidirectional(self.lstm1)
            #self.lstm2 = layers.Bidirectional(lstm2)

    def call(self, x):
        x = self.lstm1(x)
        #x = self.lstm2(x)
        return x

def createSAModel(maxlen, vocab_size, embed_dim, tokenizer, embedding_file, units=50, bidir=False, dropout_rate=0.):
    inputs = layers.Input(shape = (maxlen,))
    x = TokenAndEmbedding(vocab_size, embed_dim, tokenizer, embedding_file)(inputs)
    x = LSTMEncoder(units, bidir)(x)
    x = layers.Dense(5, activation="softmax")(x)
    outputs = layers.Dropout(dropout_rate)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

'''
class SAModel(keras.Model):
    def __init__(self, maxlen, vocab_size, embed_dim, texts, embedding_file, units=50, bidir=False, dropout_rate=0.):
        super(SAModel, self).__init__()
        
        self.tokenizerAndEmbedding = TokenAndEmbedding(maxlen, vocab_size, embed_dim, texts, embedding_file)
        self.lstmEncoder = LSTMEncoder(units, bidir)

    def call(self, inputs, training=False):
        h = self.tokenizerAndEmbedding(inputs)
        h = self.lstmEncoder(h)
        if training:
            h = layers.Dropout(dropout_rate)(h, training=training)
        h = layers.Dense(5, activation="softmax")(h)
        return Add()([inputs, h])
'''

if (__name__ == '__main__'):
    maxlen = 200
    vocab_size = 20000
    embed_dim = 50
    dataset = data_prep.load_dataset('../data/B2W-Reviews01_10000_train.csv')
    inputs = np.array([dataset["review_text"]]).T
    outputs = np.array([dataset["rating"]]).T
    tokenizer = data_prep.create_tokenizer(inputs, vocab_size, maxlen)
    tokenized_inputs = tokenizer(inputs)
    print(inputs.shape, tokenized_inputs.shape)
    word2vec_file = "../data/word2vec_200k.txt"
    embedding_layer = TokenAndEmbedding(vocab_size, embed_dim, tokenizer, word2vec_file)
    print(inputs[0], tokenized_inputs[0])
    embedded_dataset = embedding_layer(tokenized_inputs)
    print(embedded_dataset[0])
    print(embedded_dataset.shape)
    #model = SAModel(maxlen, vocab_size, embed_dim, inputs, word2vec_file, 64, False, 0.25)(inputs=input_layer)
    model = createSAModel(maxlen, vocab_size, embed_dim, tokenizer, word2vec_file, 128, False, 0.25)
    model.summary()
    print(model.predict(tokenized_inputs).shape)
