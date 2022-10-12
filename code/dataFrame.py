import streamlit as st


def init():
    # global df_train
    # global df_train_final

    if 'df_train' not in st.session_state:
        st.session_state.df_train = None

    if 'df_train_final' not in st.session_state:
        st.session_state.df_train_final = None
