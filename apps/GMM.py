from apps.Kmeans import word_cloud
from apps.TF_IDF_vect import Tfidf 
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import sys
sys.path.insert(
    0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')


def app():
    st.write('Please select the output you would like to proceed')
    col1, col2 =st.columns(2)
    with col1:
        st.session_state.word_cloud = st.checkbox('Wordcloud')

    with col2:
        st.session_state.Scatter_plot = st.checkbox('Scatter plot')

    if st.button('process'):
        st.session_state.clustering_processed = True

    if st.session_state.clustering_processed:
        # Vectorize document using TF-IDF
        # Fit and Transfrom Text Data
        X_train_counts = Tfidf(
            st.session_state.df_train['text_PP'])

        # Check Shape of Count Vector
        # checkShape = st.checkbox('check the shape of count vector')
        # if checkShape:
        #st.write(X_train_counts.shape)

        
        
        cluster_n = st.slider('Please select the number of cluster you would like to split', 0, 20, 3)
        # perform GMM clustering
        sklearn_pca = TruncatedSVD(n_components=cluster_n)
        Y_sklearn = sklearn_pca.fit_transform(X_train_counts)
        gmm = GaussianMixture(
            n_components=cluster_n, covariance_type='full').fit(Y_sklearn)
        prediction_gmm = gmm.predict(Y_sklearn)
        probs = gmm.predict_proba(Y_sklearn)

        centers = np.zeros((cluster_n, cluster_n))
        for i in range(cluster_n):
            density = mvn(cov=gmm.covariances_[
                i], mean=gmm.means_[i]).logpdf(Y_sklearn)
            centers[i, :] = Y_sklearn[np.argmax(density)]






        if st.session_state.Scatter_plot:

            plt.figure(figsize=(10, 8))
            plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],
                        c=prediction_gmm, s=50, cmap='viridis')
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.6)
            st.pyplot(plt)
        if st.session_state.word_cloud:
            st.write('You select word cloud')