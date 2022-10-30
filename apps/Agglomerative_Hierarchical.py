
import streamlit as st
import pandas as pd
from apps.TF_IDF_vect import Tfidf 
from apps.WordCloudPlt import word_cloud,display_metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import matplotlib
import plotly.figure_factory as ff
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
matplotlib.use("agg")
from PIL import Image

def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def app():

    #AgglomerativeClustering().get_params()

 
    X = st.session_state.df_train['text_PP']
    clusters_no = st.number_input('Cluster number',min_value=2,step = 1)


    if st.button('Start'):
            st.session_state.clustering_start = True

    if st.session_state.clustering_start:
        #TF-IDF -> Sparse to Dense -> Clustering
        X_train_counts = Tfidf(X)
        sklearn_prep = FunctionTransformer(lambda x: x.todense(), accept_sparse=True)
        Y_sklearn = sklearn_prep.fit_transform(X_train_counts)
        AMCLSTG=AgglomerativeClustering(linkage='average',n_clusters=clusters_no)
        model = AMCLSTG.fit(Y_sklearn)
        label = AMCLSTG.fit_predict(Y_sklearn)


        #Evaluate Clustering Performance
        display_metrics(X_train_counts,label)
        
        st.write('Please select the output you would like to proceed')
        col1, col2 =st.columns(2)
        with col1:
            word_cloud_check = st.checkbox('Wordcloud')

        with col2:
            dendrogram_check = st.checkbox('Dendrogram')
            level_no = st.number_input('Please enter the level you would like to see from the dendrogram',min_value=2,step = 1,value = 10)

        if st.button('process'):
            st.session_state.clustering_processed_3 = True

        if st.session_state.clustering_processed_3:

            if dendrogram_check:
                
                plot_dendrogram(model, labels=X.index, orientation='right',p=level_no,truncate_mode='level')
                plt.savefig('./output/AH.jpg')
                image = Image.open('./output/AH.jpg')
                st.image(image)

            if word_cloud_check:

                df=pd.DataFrame({"text":st.session_state.df_train['text_PP'],"labels":label})


                for i in df.labels.unique():
                    new_df=df[df.labels==i]
                    text="".join(new_df.text.tolist())
                    clusters_no = i+1
                    word_cloud(text,clusters_no)

