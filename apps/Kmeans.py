
from sklearn.cluster import KMeans
from apps.WordCloudPlt import word_cloud,display_metrics
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from apps.TF_IDF_vect import Tfidf 
from sklearn.decomposition import TruncatedSVD


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
        # # Compute DBI score
        # dbi = metrics.davies_bouldin_score(X_train_counts.toarray(), pred_labels)
        # dbi = round(dbi,2)

        # # Compute Silhoutte Score
        # ss = metrics.silhouette_score(X_train_counts.toarray(), pred_labels , metric='euclidean')
        # ss = round(ss,2)

        # # Print the DBI and Silhoutte Scores
        # st.write('Evaluate the clustering using Davies-Bouldin Index and Silhouette Score')
        # col1, col2 = st.columns(2)
        # col1.metric("DBI Score", dbi)
        # col2.metric("Silhoutte Score", ss)
        


        if word_cloud_check:
        
            df=pd.DataFrame({"text":st.session_state.df_train['text_PP'],"labels":pred_labels})


            for i in df.labels.unique():
                new_df=df[df.labels==i]
                text="".join(new_df.text.tolist())
                cluster_no = i+1
                word_cloud(text,cluster_no)

        if scatter_plot_check:
            sklearn_pca = TruncatedSVD(n_components=cluster_n)
            Y_sklearn = sklearn_pca.fit_transform(X_train_counts)

            kmeans_2 = KMeans(n_clusters=cluster_n).fit(Y_sklearn)

            pred_y = kmeans_2.predict(Y_sklearn)
            plt.figure(figsize = (10,8))
            plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=pred_y ,s=50, cmap='viridis')
            centroids = kmeans_2.cluster_centers_


            plt.scatter(centroids[:, 0], centroids[:, 1],c='black', s=300, alpha=0.6)
            st.pyplot(plt)











