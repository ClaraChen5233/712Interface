from apps.WordCloudPlt import word_cloud,display_metrics
from apps.TF_IDF_vect import Tfidf 
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import sys
sys.path.insert(
    0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')


def app():
    st.write('Please select the output you would like to proceed')
    col1, col2 =st.columns(2)
    with col1:
        word_cloud_check = st.checkbox('Wordcloud')

    with col2:
        scatter_plot_check = st.checkbox('Scatter plot')

    if st.button('process'):
        st.session_state.clustering_processed = True

    if st.session_state.clustering_processed:
        # Vectorize document using TF-IDF
        # Fit and Transfrom Text Data
        X_train_counts = Tfidf(
            st.session_state.df_train['text_PP'])        
        
        cluster_n = st.slider('Please select the number of cluster you would like to split', 2, 20, 3)



        # perform GMM clustering
        sklearn_pca = TruncatedSVD(n_components=cluster_n)
        Y_sklearn = sklearn_pca.fit_transform(X_train_counts)
        gmm = GaussianMixture(
            n_components=cluster_n, covariance_type='full').fit(Y_sklearn)
        prediction_gmm = gmm.predict(Y_sklearn)
        GMM_Label=prediction_gmm
        probs = gmm.predict_proba(Y_sklearn)

        centers = np.zeros((cluster_n, cluster_n))
        for i in range(cluster_n):
            density = mvn(cov=gmm.covariances_[
                i], mean=gmm.means_[i]).logpdf(Y_sklearn)
            centers[i, :] = Y_sklearn[np.argmax(density)]
        

        #Evaluate Clustering Performance
        display_metrics(X_train_counts,GMM_Label)
        
        if scatter_plot_check:

            cluster = []
            for i in prediction_gmm:
                cluster.append('cluster'+str(i+1))
            cluster_f=pd.DataFrame(cluster)
            
            legend_df = cluster_f.sort_values(by=[0], ascending=False)

            fig = px.scatter( x=Y_sklearn[:, 0], y=Y_sklearn[:, 1], color=legend_df[0])

            st.plotly_chart(fig)

        if word_cloud_check:

            df=pd.DataFrame({"text":st.session_state.df_train['text_PP'],"labels":GMM_Label})
            final_df = df.sort_values(by=['labels'], ascending=False)


            for i in final_df.labels.unique():
                new_df=df[df.labels==i]
                text="".join(new_df.text.tolist())
                cluster_no = i+1
                word_cloud(text,cluster_no)