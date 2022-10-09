import streamlit as st

import sys
sys.path.insert(0,'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
import dataFrame

dataFrame.init()

print(dataFrame.df_train)
st.markdown("""
# Machine learning app

some description

""")

