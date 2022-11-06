import sys
sys.path.insert(
    0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
import pandas as pd
from numpy import empty
import dataFrame
from attr import define
import streamlit as st




st.markdown("""
# Machine learning app

This Machine learning app allows you to do no code unsupervised machine learning💻


""")
st.subheader("""
    Here is a basic tour of the app
""")
st.write("""
🧭Navigation bar located on the left hand side of the web, you can navigate around the page using the nav bar. \n
1️⃣Please visit text processing page first to upload the dataset you would like to process, please make sure you have done text processing task before you conduct any clustering work. \n
2️⃣After text processing is done, you can navigate to clustering page using the nav bar on the left. \n
3️⃣At the clustering page, you can select different models to preform clustering, wordcloud and scatter plot are available for visualization. \n
4️⃣For Kmeans and GaussianMixture, you are able to change the number of cluster you would like to have for the final result, you will get an instant updates of the visualisation.\n
5️⃣for Agglomerative Hierarchical clustering, it takes longer time to generate the result🙂feel free to take a coffee break during the run time.\n


""")

st.subheader("""
    Q&A
""")

with st.expander("What can this app do?"):
    st.write("""
        The main features of this app is to perform Natual languague processing and used the processed data to preform clustering task.\n
        You can perform text processing including tokenization, Stopword removal and basic text cleansing; \n
        After text processing, you can go to the clustering page.\n
        There are 3 algorithms of clustering available in this app including Kmeans, GaussianMixture and Agglomerative Hierarchical clustering, \n
        you can comnpare the performance of these 3 model using the computed metrics, the app will also produce visualization of the clusters to 
        assist with understanding the data. 
    """)

with st.expander("What is Kmean"):

    st.write("""
        k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters 
        in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.\n
        k-means clustering minimizes within-cluster variances\n
    """)
with st.expander("What is GaussianMixture"):

    st.write("""
        GMMs are a generalization of Gaussian distributions and can be used to represent any data set that can be clustered into 
        multiple Gaussian distributions. The Gaussian mixture model is a probabilistic model that assumes all the data points are 
        generated from a mix of Gaussian distributions with unknown parameters. 
    """)
with st.expander("What is Agglomerative Hierarchical clustering"):

    st.write("""
The agglomerative clustering is the most common type of hierarchical clustering used to group objects in clusters based on their similarity. 
It’s also known as AGNES (Agglomerative Nesting). The algorithm starts by treating each object as a singleton cluster. Next, pairs of clusters 
are successively merged until all clusters have been merged into one big cluster containing all objects. The result is a tree-based representation
 of the objects, named dendrogram.
    """)
#Home = MultiApp()
#dataFrame.init()


#Home.add_app("Text mining", Text_processing.app)
#Home.add_app("Pattern mining", UploadFile.app)

#Home.run()
