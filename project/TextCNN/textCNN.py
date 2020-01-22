
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model, model_from_json
from keras.callbacks import Callback
from keras.layers import Input, Embedding, Dense, GlobalMaxPool1D, Reshape, Conv2D, MaxPool2D, \
    Flatten, Concatenate, Dropout, SpatialDropout1D, BatchNormalization, Activation
from keras.preprocessing import text, sequence
from keras.utils import plot_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utility import clean


# model definition
def textCNN(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    input = Input(shape=(Config.max_sentence_len,))
    x = Embedding(Config.max_features, Config.embedding_size, weights=[embedding_matrix])(input)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((Config.max_sentence_len, Config.embedding_size, 1))(x)

    conv_0 = Conv2D(Config.num_filters, kernel_size=(filter_sizes[0], Config.embedding_size), kernel_initializer='normal', activation='relu')(x)
    batchnorm_0 = BatchNormalization()(conv_0)
    maxpool_0 = MaxPool2D(pool_size=(Config.max_sentence_len - filter_sizes[0] + 1, 1))(batchnorm_0)

    conv_1 = Conv2D(Config.num_filters, kernel_size=(filter_sizes[1], Config.embedding_size), kernel_initializer='normal', activation='relu')(x)
    batchnorm_1 = BatchNormalization()(conv_1)
    maxpool_1 = MaxPool2D(pool_size=(Config.max_sentence_len - filter_sizes[1] + 1, 1))(batchnorm_1)

    conv_2 = Conv2D(Config.num_filters, kernel_size=(filter_sizes[2], Config.embedding_size), kernel_initializer='normal', activation='relu')(x)
    batchnorm_2 = BatchNormalization()(conv_2)
    maxpool_2 = MaxPool2D(pool_size=(Config.max_sentence_len - filter_sizes[2] + 1, 1))(batchnorm_2)

    conv_3 = Conv2D(Config.num_filters, kernel_size=(filter_sizes[3], Config.embedding_size), kernel_initializer='normal', activation='relu')(x)
    batchnorm_3 = BatchNormalization()(conv_3)
    maxpool_3 = MaxPool2D(pool_size=(Config.max_sentence_len - filter_sizes[3] + 1, 1))(batchnorm_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)
    output = Dense(6, activation="sigmoid")(z)

    model = Model(inputs=input, outputs=output)

    return model


# roc evaluation metric
class auc_eval(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.val_features, self.val_labels = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.val_features, verbose=0)
            print("\n AUC score: %.8f \n" % (roc_auc_score(self.val_labels, y_pred)))


# Config
class Config:
    max_features = 100000
    max_sentence_len = 200
    embedding_size = 300
    num_filters = 32
    batch_size = 256
    epochs = 4
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def data_prep(df_train, df_test, fast_text_embeddings):

    def extract_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    # cleaning the comment text
    df_train['comment_text'] = df_train['comment_text'].apply(lambda x: clean(x))
    df_test['comment_text'] = df_test['comment_text'].apply(lambda x: clean(x))

    # Handling the nan values
    train_features = df_train["comment_text"].fillna("fillna").values
    train_labels = df_train[Config.labels].values
    test_features = df_test["comment_text"].fillna("fillna").values

    # data preprocessing
    tokenizer = text.Tokenizer(num_words=Config.max_features)
    tokenizer.fit_on_texts(list(train_features) + list(test_features))
    train_features = tokenizer.texts_to_sequences(train_features)
    test_features = tokenizer.texts_to_sequences(test_features)
    train_features = sequence.pad_sequences(train_features, maxlen=Config.max_sentence_len)
    test_features = sequence.pad_sequences(test_features, maxlen=Config.max_sentence_len)

    # creating the embedding matrix
    embeddings_index = dict(extract_coefs(*emb.rstrip().rsplit(' ')) for emb in open(fast_text_embeddings))
    word_index = tokenizer.word_index
    nb_words = min(Config.max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, Config.embedding_size))

    for word, idx in word_index.items():
        if idx >= Config.max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[idx] = embedding_vector

    return train_features, train_labels, test_features, embedding_matrix


def load_create_model(model_json_file="model.json", model_weights_file="textCNN.h5"):
    # load json and create model
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights_file)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def main():
    np.random.seed(42)

    # loading the data
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    df_submission = pd.read_csv('../data/sample_submission.csv')
    fast_text_embeddings = '../data/crawl-300d-2M.vec'

    train_features, train_labels, test_features, embedding_matrix = data_prep(df_train, df_test, fast_text_embeddings)

    # creating the model
    model = textCNN(embedding_matrix=embedding_matrix)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model = load_create_model()
    model.summary()


    # plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

    # train val split
    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, train_size=0.95, random_state=225)

    # creating the metric
    auc = auc_eval(validation_data=(val_features, val_labels), interval=1)

    # training the model
    hist = model.fit(train_features,
                     train_labels,
                     batch_size=Config.batch_size,
                     epochs=Config.epochs,
                     verbose=1,
                     validation_data=(val_features, val_labels),
                     callbacks=[auc]
                     )

    # saving the model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    print("Saved model to disk")
    model.save_weights("textCNN.h5")

    # testing and getting predictions
    predicted_labels = model.predict(test_features, batch_size=1024, verbose=1)

    # plotting the loss plots
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('textCNN_loss.png')

    # saving the history
    import json
    with open('textCNN_hist.json', 'w') as f:
        json.dump(hist.history, f)

    # generating the submission
    df_submission[Config.labels] = predicted_labels
    df_submission.to_csv('textCNN_submission.csv', index=False)


if __name__ == '__main__':
    main()