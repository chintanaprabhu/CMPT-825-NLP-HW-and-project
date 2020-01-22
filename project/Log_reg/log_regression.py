import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

import sys
sys.path.append('../')
from utility import clean


# Config
class Config:
    data_path = '../data/'
    word_vec_max_features = 25000
    char_vec_max_features = 35000
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    iterations = 200


def read_data():
    df_train = pd.read_csv(Config.data_path+'train.csv')
    df_test = pd.read_csv(Config.data_path+'test.csv')

    df_train['comment_text_clean']=df_train['comment_text'].apply(lambda x :clean(x))
    df_test['comment_text_clean']=df_test['comment_text'].apply(lambda x :clean(x))

    df_test.fillna(' ',inplace=True)
    return(df_train, df_test)


def prep_model(df_train, df_test):

    word_vec = TfidfVectorizer(max_features=Config.word_vec_max_features,analyzer='word',ngram_range=(1,3),dtype=np.float32)
    char_vec = TfidfVectorizer(max_features=Config.char_vec_max_features,analyzer='char',ngram_range=(3,6),dtype=np.float32)

    # Word n-gram vector
    train_vec = word_vec.fit_transform(df_train['comment_text_clean'])
    test_vec = word_vec.transform(df_test['comment_text_clean'])

    # Character n-gram vector
    train_vec_char = char_vec.fit_transform(df_train['comment_text_clean'])
    test_vec_char = char_vec.transform(df_test['comment_text_clean'])

    train = sparse.hstack([train_vec, train_vec_char])
    test = sparse.hstack([test_vec, test_vec_char])

    y = df_train[Config.labels]
    del train_vec, test_vec, train_vec_char, test_vec_char

    res = np.zeros((test.shape[0],y.shape[1]))
    cv_score =[]
    for i,col in enumerate(Config.labels):
        log_reg = LogisticRegression(class_weight = 'balanced', max_iter=Config.iterations)
        print('Building {} model for :{''}'.format(i, col))
        log_reg.fit(train, y[col])
        res[:,i] = log_reg.predict_proba(test)[:, 1]

    res_1 = pd.DataFrame(res,columns=y.columns)
    submit = pd.concat([df_test['id'], res_1],axis=1)
    submit.to_csv('logReg_submission.csv', index=False)

    return(submit, log_reg, train, test)


def pred_lr(log_reg, train):
    pred = log_reg.predict(train)
    return(pred)


if __name__ == '__main__':
    df_train, df_test = read_data()
    submit, log_reg, train, test = prep_model(df_train, df_test)
