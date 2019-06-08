import pandas as pd
import spacy
import time
import gzip
import dill

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import STOP_WORDS
from html import unescape


nlp = spacy.load('en')
STOP_WORDS_lemma = set([word.lemma_ for
                        word in nlp(' '.join(list(STOP_WORDS)))])
STOP_WORDS_lemma = STOP_WORDS_lemma.union({',', '.', ';'})

# Handling html tags


def preprocessor(doc):
    return unescape(doc).lower


def lemmatizer(doc):
    return [word.lemma_ for word in nlp(doc)]


def fit_model(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        model = func()
        model.fit(X_train, y_train)
        t2 = time.time() - t1

        print('Training took: {}'.format(t2))
        print('Training accuracy: {}'.format(model.score(X_train, y_train)))
        print('Testing accuracy: {}'.format(model.score(X_test, y_test)))
        return model
    return wrapper


@fit_model
def construct_sentiment_model():
    vectorizer = TfidfVectorizer(preprocessor=preprocessor,
                                 tokenizer=lemmatizer,
                                 ngram_range=(1, 2),
                                 stop_words=STOP_WORDS_lemma)
    clf = MultinomialNB()
    pipe = Pipeline([('vectorizer', vectorizer),
                     ('classifier', clf)])
    return pipe


if __name__ == "__main__":
    df = pd.read_csv('data/Sentiment-Analysis-Dataset.csv',
                     error_bad_lines=False, engine='python')
    X = df['SentimentText']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = construct_sentiment_model()

    # serialize model
    with gzip.open('sentiment_model.dill.gz', 'wb') as f:
        dill.dump(model, f, recurse=True)
