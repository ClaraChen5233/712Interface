import streamlit as st
from multiapp import MultiApp
import os
import sys

sys.path.insert(0,'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
import dataFrame
from apps import GMM # import app modules 

Clustering = MultiApp()

st.markdown("""
This is clustering page
""")
if dataFrame.df_train is None:
    st.write('Please upload a file to do preprocessing first')
else:
    
    # Add all your application here
    Clustering.add_app("GMM", GMM.app())

    # The main app
    Clustering.run()
