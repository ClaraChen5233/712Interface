import sys
sys.path.insert(
    0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
import pandas as pd
from numpy import empty
import dataFrame
from attr import define
import streamlit as st
from multiapp import MultiApp
from apps import Text_processing



st.markdown("""
# Machine learning app

This Machine learning app allows you to do free code unsupervised machine learning

Please choose a machine learning task from below:

""")


Home = MultiApp()
dataFrame.init()


Home.add_app("Text mining", Text_processing.app)
#Home.add_app("Pattern mining", UploadFile.app)

Home.run()
