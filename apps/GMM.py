# TF-IDF Feature Generation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
import dataFrame


def app():
    # Initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # # Vectorize document using TF-IDF
    tf_idf_vect = TfidfVectorizer(lowercase=True,
                            stop_words='english',
                            ngram_range = (1,1),
                            tokenizer = tokenizer.tokenize)



    # Fit and Transfrom Text Data
    X_train_counts = tf_idf_vect.fit_transform(dataFrame.df_train['text_PP'])

    # Check Shape of Count Vector
    # checkShape = st.checkbox('check the shape of count vector')
    # if checkShape:
    st.write(X_train_counts.shape)

    #perform GMM clustering   
    sklearn_pca = TruncatedSVD(n_components = 2)
    Y_sklearn = sklearn_pca.fit_transform(X_train_counts)
    gmm = GaussianMixture(n_components=3, covariance_type='full').fit(Y_sklearn)
    prediction_gmm = gmm.predict(Y_sklearn)
    probs = gmm.predict_proba(Y_sklearn)

    centers = np.zeros((3,2))
    for i in range(3):
        density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(Y_sklearn)
        centers[i, :] = Y_sklearn[np.argmax(density)]

    plt.figure(figsize = (10,8))
    plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6)