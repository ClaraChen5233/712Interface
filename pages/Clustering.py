from sklearn.cluster import KMeans
from apps import GMM,Kmeans,Agglomerative_Hierarchical  # import app modules
import dataFrame
import streamlit as st
from multiapp import MultiApp
import os
import sys

sys.path.insert(
    0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')

Clustering = MultiApp()
# st.write(st.session_state.df_train['text_PP'])

st.markdown("""
This is clustering page
""")
if st.session_state.df_train_final is None:
    st.write('Please upload a file to do preprocessing first')

else:

    if st.session_state.df_train_final is not None:

        # Add all your application here
        Clustering.add_app("GMM", GMM.app)
        Clustering.add_app("Kmeans", Kmeans.app)
        Clustering.add_app("Agglomerative Hierarchical", Agglomerative_Hierarchical.app )

        # The main app
        Clustering.run()

    # except:
    #     st.write('Please upload a file to do preprocessing first')
