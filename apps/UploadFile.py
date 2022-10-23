import streamlit as st
import pandas as pd

def app():

    st.subheader('Please upload a file to do the pre-processing')
    st.session_state.uploaded_file = st.file_uploader(
            "Choose a file", type=['csv'], accept_multiple_files=False)

    if st.session_state.uploaded_file is not None:
        df_train_orgin = st.session_state.df_train
        st.session_state.df_train = pd.read_csv(st.session_state.uploaded_file)
        display = st.checkbox('Display data frame')
        if display:
            st.write(df_train_orgin)
