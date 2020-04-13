from enum import Enum
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics

import sklearn_crfsuite

import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class Tweet:
    def __init__(self, id, subject, tweettext):
        self.id = id
        self.subject = subject
        self.tweettext = tweettext


def get_all_tweets():
    train_set = list(get_tweets("resources/TweetsTrainset.txt"))
    test_set = list(get_tweets("resources/TweetsTestset.txt"))
    ground_truth = list(get_tweets("resources/TweetsTestGroundTruth.txt"))
    return train_set, test_set, ground_truth


def get_tweets(dataset_name):
    with open(dataset_name) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    for line in content:
        parts = line.split('\t')
        id = int(parts[0])
        if len(parts) == 3:
            subjects = parse_subjects(parts[1])
            tweettext = parts[2]
        elif len(parts) == 2:
            subjects = None
            tweettext = parts[1]
        else:
            raise ValueError(f"Error in line {line}")
        yield Tweet(id, subjects, tweettext)

def parse_subjects(part):
    subjects = part.split(';')
    subjects = [s for s in subjects if s != '']
    subjects = [s.split('/')[1] for s in subjects]

    return subjects


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in words]


def get_raw_tweets(train, test):
    X_train = [t.tweettext for t in train]
    X_test = [t.tweettext for t in test]
    return X_train, X_test


def get_subjects(train, ground_truth):
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform([t.subject for t in train])
    y_test = mlb.transform([t.subject for t in ground_truth])
    return y_train, y_test


def get_dtms(X_train, X_test):
    stop_words = tokenize(' '.join(stopwords.words("english")))
    vec = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    X_train_dtm = vec.fit_transform(X_train)
    X_test_dtm = vec.transform(X_test)
    return X_train_dtm, X_test_dtm


def predict(X_train_dtm, X_test_dtm, y_train, random_state=42, cls=DecisionTreeClassifier):
    cl = cls()
    print(f"Using classifier: {repr(cl)}")
    cl.fit(X_train_dtm, y_train)
    return cl.predict(X_test_dtm)


def evaluate(y_test, y_test_pred):
    prec = metrics.precision_score(y_test, y_test_pred, average='micro')
    recall = metrics.recall_score(y_test, y_test_pred, average='micro')
    f1 = metrics.f1_score(y_test, y_test_pred, average='micro')

    print(f"Precision: {prec:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")


def main():
    train_set, test_set, ground_truth = get_all_tweets()

    X_train, X_test = get_raw_tweets(train_set, test_set)
    y_train, y_test = get_subjects(train_set, ground_truth)

    X_train_dtm, X_test_dtm = get_dtms(X_train, X_test)

    print(X_train_dtm.shape)
    print(X_test_dtm.shape)
    print(y_train.shape)
    print(y_test.shape)

    from sklearn.naive_bayes import MultinomialNB
    y_test_pred = predict(X_train_dtm, X_test_dtm, y_train, cls=MultinomialNB)
    evaluate(y_test, y_test_pred)

if __name__ == '__main__':
    main()
