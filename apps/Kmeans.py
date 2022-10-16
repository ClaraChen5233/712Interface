
import numpy as np
# Import KMeans Model
from sklearn.cluster import KMeans
from sklearn import metrics
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from apps.TF_IDF_vect import Tfidf 
from sklearn.decomposition import TruncatedSVD

# import sys
# sys.path.insert(
#     0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
def word_cloud(text,cluster_no):
    # Create stopword list
    stopword_list = set(STOPWORDS) 

    # Create WordCloud 
    word_cloud = WordCloud(width = 800, height = 500, 
                        background_color ='white', 
                        stopwords = stopword_list, 
                        min_font_size = 14).generate(text) 

    # Set wordcloud figure size
    plt.figure(figsize = (8, 6)) 
    
    # Set title for word cloud
    plt.title('Cluster Number: '+ str(cluster_no))
    
    # Show image
    plt.imshow(word_cloud) 

    # Remove Axis
    plt.axis("off")  

    # save word cloud
    # plt.savefig(wc_file_name,bbox_inches='tight')

    # show plot
    st.pyplot(plt)


def app():

    def word_cloud(text,cluster_no):
        # Create stopword list
        stopword_list = set(STOPWORDS) 

        # Create WordCloud 
        word_cloud = WordCloud(width = 800, height = 500, 
                            background_color ='white', 
                            stopwords = stopword_list, 
                            min_font_size = 14).generate(text) 

        # Set wordcloud figure size
        plt.figure(figsize = (8, 6)) 
        
        # Set title for word cloud
        plt.title('Cluster Number: '+ str(cluster_no))
        
        # Show image
        plt.imshow(word_cloud) 

        # Remove Axis
        plt.axis("off")  

        # save word cloud
        # plt.savefig(wc_file_name,bbox_inches='tight')

        # show plot
        st.pyplot(plt)

    
    # Vectorize document using TF-IDF
    # Fit and Transfrom Text Data
    X_train_counts = Tfidf(
        st.session_state.df_train['text_PP'])

    # Check Shape of Count Vector
    # checkShape = st.checkbox('check the shape of count vector')
    # if checkShape:
    st.write(X_train_counts.shape)


    #perform k-means clustering
    #prompt user enter cluster number 
    cluster_no = st.text_input(label ='Please enter the number of cluster', placeholder='2')
    if cluster_no !='':
        # Create Kmeans object and fit it to the training data 
        kmeans = KMeans(n_clusters=2).fit(X_train_counts)

        # Get the labels using KMeans
        pred_labels = kmeans.labels_


        #Evaluate Clustering Performance
        # Compute DBI score
        dbi = metrics.davies_bouldin_score(X_train_counts.toarray(), pred_labels)

        # Compute Silhoutte Score
        ss = metrics.silhouette_score(X_train_counts.toarray(), pred_labels , metric='euclidean')

        # Print the DBI and Silhoutte Scores
        st.write('Evaluate the clustering using Davies-Bouldin Index and Silhouette Score')
        st.write("DBI Score: ", dbi, "\nSilhoutte Score: ", ss)


        
        df=pd.DataFrame({"text":st.session_state.df_train['text_PP'],"labels":pred_labels})


        for i in df.labels.unique():
            new_df=df[df.labels==i]
            text="".join(new_df.text.tolist())
            cluster_no = i+1
            word_cloud(text,cluster_no)

















    # # perform GMM clustering
    # sklearn_pca = TruncatedSVD(n_components=2)
    # Y_sklearn = sklearn_pca.fit_transform(X_train_counts)
    # gmm = GaussianMixture(
    #     n_components=3, covariance_type='full').fit(Y_sklearn)
    # prediction_gmm = gmm.predict(Y_sklearn)
    # probs = gmm.predict_proba(Y_sklearn)

    # centers = np.zeros((3, 2))
    # for i in range(3):
    #     density = mvn(cov=gmm.covariances_[
    #         i], mean=gmm.means_[i]).logpdf(Y_sklearn)
    #     centers[i, :] = Y_sklearn[np.argmax(density)]


    # plt.figure(figsize=(10, 8))
    # plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],
    #             c=prediction_gmm, s=50, cmap='viridis')
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.6)
    # st.pyplot(plt)
