
from sklearn.cluster import KMeans
from apps.WordCloudPlt import word_cloud,display_metrics
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from apps.TF_IDF_vect import Tfidf 
from sklearn.decomposition import TruncatedSVD
import plotly.express as px



def app():

    st.write('Please select the output you would like to proceed')
    col1, col2 =st.columns(2)
    with col1:
        word_cloud_check = st.checkbox('Wordcloud')

    with col2:
        scatter_plot_check = st.checkbox('Scatter plot')

    if st.button('process'):
        st.session_state.clustering_processed_2 = True

    if st.session_state.clustering_processed_2:
        # Vectorize document using TF-IDF
        # Fit and Transfrom Text Data
        X_train_counts = Tfidf(
            st.session_state.df_train['text_PP'])


        #perform k-means clustering
        #prompt user enter cluster number 
        cluster_n = st.slider('Please select the number of cluster you would like to split', 2, 20, 3)
        
        # Create Kmeans object and fit it to the training data 
        kmeans = KMeans(n_clusters=cluster_n).fit(X_train_counts)

        # Get the labels using KMeans
        pred_labels = kmeans.labels_


        #Evaluate Clustering Performance
        display_metrics(X_train_counts,pred_labels)


        if word_cloud_check:
        
            df=pd.DataFrame({"text":st.session_state.df_train['text_PP'],"labels":pred_labels})
            final_df = df.sort_values(by=['labels'], ascending=True)


            for i in final_df.labels.unique():
                new_df=final_df[df.labels==i]
                text="".join(new_df.text.tolist())
                cluster_no = i+1
                word_cloud(text,cluster_no)

        if scatter_plot_check:
            sklearn_pca = TruncatedSVD(n_components=cluster_n)
            Y_sklearn = sklearn_pca.fit_transform(X_train_counts)

            kmeans_2 = KMeans(n_clusters=cluster_n).fit(Y_sklearn)

            pred_y = kmeans_2.predict(Y_sklearn)

            cluster = []
            for i in pred_y:
                cluster.append('cluster'+str(i+1))
            cluster_f=pd.DataFrame(cluster)

            legend_df = cluster_f.sort_values(by=[0], ascending=True)

            fig = px.scatter( x=Y_sklearn[:, 0], y=Y_sklearn[:, 1], color=legend_df[0])

            st.plotly_chart(fig)












