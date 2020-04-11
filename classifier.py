from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn import metrics

import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class Attribute(Enum):
    LOC = 1
    PER = 2
    ORG = 3
    MISC = 4
    EMPTY = 5

    def __str__(self):
        return self.value

class Tweet:
    def __init__(self, id, subject, tweettext):
        self.id = id
        self.subject = subject
        self.tweettext = tweettext

def to_attr(string):
    string = string.upper()
    if string == 'LOC':
        return Attribute.LOC
    if string == 'PER':
        return Attribute.PER
    if string == 'ORG':
        return Attribute.ORG
    if string == 'MISC':
        return Attribute.MISC
    if string == '':
        return Attribute.EMPTY
    raise ValueError(f"Fuuuuu, could not parse {string}")

def get_tweets(dataset_name):
    retval = []
    with open(dataset_name) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            parts = line.split('\t')
            id = int(parts[0])
            if len(parts) == 3:
                subjects = parts[1].split(';')
                subjects = [s for s in subjects if s != '']
                tweettext = parts[2]
                retval.append(Tweet(id, subjects, tweettext))
            elif len(parts) == 2:
                tweettext = parts[1]
                retval.append(Tweet(id, None, tweettext))
            else:
                raise ValueError(f"Error in line: {line}")
    return retval


def evaluate(y_test, y_test_pred):
    prec = metrics.precision_score(y_test, y_test_pred, average='micro')
    f1 = metrics.f1_score(y_test, y_test_pred, average='micro')
    recall = metrics.recall_score(y_test, y_test_pred, average='micro')

    print(f"Precision: {prec:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in words]


def main():
    train_set = get_tweets("TweetsTrainset.txt")
    test_set = get_tweets("TweetsTestset.txt")
    ground_truth = get_tweets("TweetsTestGroundTruth.txt")

    X_train = [t.tweettext for t in train_set]
    X_test = [t.tweettext for t in test_set]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform([t.subject for t in train_set])
    y_test = mlb.transform([t.subject for t in ground_truth])

    stop_words = tokenize(' '.join(stopwords.words("english")))
    vec = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    X_train_dtm = vec.fit_transform(X_train)
    X_test_dtm = vec.transform(X_test)

    from sklearn.tree import ExtraTreeClassifier as cls
    from sklearn.dummy import DummyClassifier as cls
    from sklearn.ensemble import ExtraTreesClassifier as cls
    cl = cls(random_state=42)
    cl.fit(X_train_dtm, y_train)
    y_test_pred = cl.predict(X_test_dtm)

    print(f"Using classifier {repr(cl)}")
    evaluate(y_test, y_test_pred)

if __name__ == '__main__':
    main()
