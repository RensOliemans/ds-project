import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics



nltk.download('reuters')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Get X_train and X_test
documents = reuters.fileids()
training_doc_ids = [f_id for f_id in documents if 'training' in f_id]
test_doc_ids = [f_id for f_id in documents if 'test' in f_id]

X_train = [reuters.raw(d) for d in training_doc_ids]
X_test = [reuters.raw(d) for d in test_doc_ids]

# Get y_train and y_test
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform([reuters.categories(d)
                             for d in training_doc_ids])
y_test = mlb.transform([reuters.categories(d)
                        for d in test_doc_ids])

# Preprocessing
def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in words]

stop_words = tokenize(' '.join(stopwords.words("english")))


v1 = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
X_train_dtm = v1.fit_transform(X_train)
X_test_dtm = v1.transform(X_test)


cl = OneVsRestClassifier(LinearSVC(random_state=42))
cl.fit(X_train_dtm, y_train)
y_test_pred = cl.predict(X_test_dtm)

# Evaluate
acc = metrics.accuracy_score(y_test, y_test_pred)
prec = metrics.precision_score(y_test, y_test_pred, average='micro')

print(f"Precision: {prec:.2%}. Accuracy: {acc:.2%}")
