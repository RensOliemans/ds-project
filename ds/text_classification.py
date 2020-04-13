from read_data import DataReader
from helper import tokenize, stop_words

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


class Classification:
    def __init__(self):
        self._setup()

    def _setup(self):
        pass

    def start(self):
        self._get_data()
        self._preprocess()
        self._predict()
        self.evaluate()

    def _get_data(self):
        print("Getting Data..")
        d = DataReader()
        self.train = d.train
        self.test = d.test
        self.ground_truth = d.ground_truth

    def _preprocess(self):
        print("Preprocessing..")
        self._set_x()
        # self._set_y()
        self._convert_x()

    def _set_x(self):
        self.x_train = self.train.tweets
        self.x_test = self.test.tweets

        self.y_train = self.train.subjects
        self.y_test = self.ground_truth.subjects

    def _set_y(self):
        mlb = MultiLabelBinarizer()
        self.y_train = mlb.fit_transform(self.train.subjects)
        self.y_test = mlb.transform(self.ground_truth.subjects)

    def _convert_x(self):
        vec = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
        self.x_train = vec.fit_transform(self.x_train)
        self.x_test = vec.transform(self.x_test)

    def _predict(self):
        print("Predicting..")
        # self._print_shapes()
        self.cl = self._get_cl()
        self.cl.fit(self.x_train, self.y_train)
        self.y_test_pred = self.cl.predict(self.x_test)

    def _print_shapes(self):
        print(self.x_train.shape)
        print(self.x_test.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)

    def _get_cl(self):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC
        return OneVsRestClassifier(LinearSVC())

    def evaluate(self):
        if not hasattr(self, 'y_test_pred'):
            print("classification.start() was not run yet, performing now")
            return self.start()

        print(f"Predicted with classifier {self.cl}")
        precision = metrics.precision_score(self.y_test, self.y_test_pred,
                                            average='micro')
        recall = metrics.recall_score(self.y_test, self.y_test_pred,
                                      average='micro')
        f1 = metrics.f1_score(self.y_test, self.y_test_pred, average='micro')

        print(f"{'Precision:'.ljust(10)} {precision:.2%}")
        print(f"{'Recall:'.ljust(10)} {recall:.2%}")
        print(f"{'F1:'.ljust(10)} {f1:.2%}")


def main():
    c = Classification()
    c.start()



if __name__ == '__main__':
    main()
