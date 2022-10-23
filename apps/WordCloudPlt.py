import streamlit as st
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt

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