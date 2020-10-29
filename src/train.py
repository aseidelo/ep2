import numpy as np
import tensorflow as tf
import model
import data_prep
import matplotlib.pyplot as plt
import argparse

def plot_metrics(history, dataset, bidir, dropout):
    metrics =  ['accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()
        if(bidir):
            plt.savefig('../doc/' + dataset + 'bidirectional_lstm_dp' + str(int(dropout*100)) + '.png')
        else:
            plt.savefig('../doc/' + dataset + 'unidirectional_lstm_dp' + str(int(dropout*100)) + '.png')
        plt.close()

def train(max_epochs, batch_size, saModel, train_inputs, train_outputs, validation_inputs, validation_outputs, dataset, bidir, dropout):
    # definir early stopping baseado na AUC
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        verbose=1,
        patience=50,
        restore_best_weights=True)

    saModel.compile("adam", "sparse_categorical_crossentropy", metrics=['accuracy'])
    history = saModel.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=max_epochs, validation_data=(validation_inputs, validation_outputs), callbacks=[early_stopping])
    plot_metrics(history, dataset, bidir, dropout)

def test(saModel, x_test, y_test, dataset, bidir, dropout):
    out_file = '../doc/' + dataset + '.txt'
    output = saModel.evaluate(x=x_test, y=y_test)
    with open(out_file, 'a+') as file:
        if(bidir):
           file.write('bidirectional lstm | dp = 0.' + str(int(dropout*100)) + '\n')
        else:
           file.write('unidirectional lstm | dp = 0.' + str(int(dropout*100)) + '\n')
        file.write('test loss: ' + str(output[0]) + '\n')
        file.write('test accuracy: ' + str(output[1]) + '\n')

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Prepare B2W dataset for Sentiment Analisys task in portuguese.')
    parser.add_argument('--inpath', metavar='<dir>', type=str, help='raw data path', default='../data/')
    parser.add_argument('--outpath', metavar='<dir>', type=str, help='path for processed data', default='../data/')
    parser.add_argument('--dataset', metavar='<csv file prefix>', type=str, help='csv dataset file name', default='B2W-Reviews01_132289')
    parser.add_argument('--epochs', metavar='N', type=int, help='Number of epochs for training', default=100)
    parser.add_argument('--maxlen', metavar='N', type=int, help='max number of tokens to be read on each sentence', default=200)
    parser.add_argument('--vocabsize', metavar='N', type=int, help='number of words to be considered for the vocabulary', default=20000)
    parser.add_argument('--embeddim', metavar='N', type=int, help='embedding size', default=50)
    parser.add_argument('--batchsize', metavar='N', type=int, help='batch size', default=32)
    parser.add_argument('--dropout', metavar='N', type=float, help='Dropout rate', default=0.0)
    parser.add_argument('-bidirectional', action='store_true', help='Use bidirectional RNN layer?')
    parser.add_argument('--word2vec', metavar='<txt file prefix>', type=str, help='embedding size', default='word2vec_200k')
    args = parser.parse_args()
    max_epochs = args.epochs
    maxlen = args.maxlen
    vocab_size = args.vocabsize
    embed_dim = args.embeddim
    dropout = args.dropout
    bidir = args.bidirectional
    dataset_train, dataset_val, dataset_test = data_prep.load_train_val_test(args.inpath + args.dataset)
    inputs_train = np.array([dataset_train["review_text"]]).T
    outputs_train = np.array([dataset_train["rating"]]).T
    inputs_val = np.array([dataset_val["review_text"]]).T
    outputs_val = np.array([dataset_val["rating"]]).T
    inputs_test = np.array([dataset_test["review_text"]]).T
    outputs_test = np.array([dataset_test["rating"]]).T
    tokenizer = data_prep.create_tokenizer(inputs_train, vocab_size, maxlen)
    tokenized_inputs_train = tokenizer(inputs_train)
    tokenized_inputs_val = tokenizer(inputs_val)
    tokenized_inputs_test = tokenizer(inputs_test)
    word2vec_file = args.inpath + args.word2vec + '.txt'
    dps = [0.1]#, 0.25, 0.5]
    bidirs = [True]#False, True]
    for dp in dps:
        for bd in bidirs:
            saModel = model.createSAModel(maxlen, vocab_size, embed_dim, tokenizer, word2vec_file, 128, bd, dp)
            train(max_epochs, args.batchsize, saModel, tokenized_inputs_train, outputs_train, tokenized_inputs_val, outputs_val, args.dataset, bd, dp)
            test(saModel, tokenized_inputs_test, outputs_test, args.dataset, bd, dp)