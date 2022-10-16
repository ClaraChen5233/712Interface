# TF-IDF Feature Generation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer


def Tfidf(df_train):
    # Initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # # Vectorize document using TF-IDF
    tf_idf_vect = TfidfVectorizer(lowercase=True,
                            stop_words='english',
                            ngram_range = (1,1),
                            tokenizer = tokenizer.tokenize)

    # Fit and Transfrom Text Data
    X_train_counts = tf_idf_vect.fit_transform(df_train)


    return(X_train_counts)
