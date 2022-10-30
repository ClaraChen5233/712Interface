import streamlit as st


def init():
    # global df_train
    # global df_train_final

    if 'df_train' not in st.session_state:
        st.session_state.df_train = None

    if 'df_train_final' not in st.session_state:
        st.session_state.df_train_final = None

    if 'process_tweets' not in st.session_state:
        st.session_state.process_tweets = False

    if 'tweets' not in st.session_state:
        st.session_state.tweets = False

    if 'tokenize' not in st.session_state:
        st.session_state.tokenize = False

    if 'clean_NonAscii' not in st.session_state:
        st.session_state.clean_NonAscii = False   

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None     

    if 'processed' not in st.session_state:
        st.session_state.processed = False

    if 'clustering_processed' not in st.session_state:
        st.session_state.clustering_processed = False

    if 'clustering_processed_2' not in st.session_state:
        st.session_state.clustering_processed_2 = False
    if 'clustering_processed_3' not in st.session_state:
        st.session_state.clustering_processed_3 = False

    if 'clustering_start' not in st.session_state:
        st.session_state.clustering_start = False

    if 'word_cloud' not in st.session_state:
        st.session_state.word_cloud= False

    if 'Scatter_plot' not in st.session_state:
        st.session_state.Scatter_plot= False

    if 'ShowTopFreq' not in st.session_state:
        st.session_state.ShowTopFreq= False

    if 'message'not in st.session_state:
        st.session_state.message= False


