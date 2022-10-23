import streamlit as st
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from sklearn import metrics

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


def display_metrics(X_train_counts,pred_labels):
    
    #Evaluate Clustering Performance
    # Compute DBI score
    dbi = metrics.davies_bouldin_score(X_train_counts.toarray(), pred_labels)
    dbi = round(dbi,2)

    # Compute Silhoutte Score
    ss = metrics.silhouette_score(X_train_counts.toarray(), pred_labels , metric='euclidean')
    ss = round(ss,2)

    # Print the DBI and Silhoutte Scores
    st.write('Evaluate the clustering using Davies-Bouldin Index and Silhouette Score')
    col1, col2 = st.columns(2)
    col1.metric("DBI Score", dbi)
    col2.metric("Silhoutte Score", ss)