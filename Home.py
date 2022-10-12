import sys
sys.path.insert(
    0, 'C:/Users/User/Desktop/multi-page-app-main/multi-page-app-main/code')
from text_processing_code import *
from cleanTweets import CleanTweets
import pandas as pd
from numpy import empty
import dataFrame
from attr import define
import streamlit as st



dataFrame.init()

st.markdown("""
# Machine learning app

some description

""")


st.subheader('Please upload a file to do the pre-processing')
CT = CleanTweets()


def confirm_checkbox(process_tweets, tweets, tokenize, clean_NonAscii, df_train, df_column):
    df_column_name = df_column
    if process_tweets:
        df_train = CT.clean_process(df_train, df_column)

    if tweets:
        df_train = pre_processing(df_train)

    if tokenize:
        df_train = text_tokenize(df_train)

    if clean_NonAscii:
        df_train = CT.clean_NA_process(df_train, df_column_name)
        df_column_name = 'text_PP'

    message = True

    return message, df_train


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


uploaded_file = st.file_uploader(
    "Choose a file", type=['csv'], accept_multiple_files=False)
if 'processed' not in st.session_state:
    st.session_state.processed = False

if uploaded_file is not None:
    st.session_state.processed = False
    # read the uploaded file

    st.session_state.df_train = pd.read_csv(uploaded_file)
    # save an orignial version
    df_train_orgin = st.session_state.df_train
    # give an option when user would like to display the data
    display = st.checkbox('Display data frame')
    if display:
        st.write(df_train_orgin)

    if df_train_orgin.columns is not None:
        # dataFrame.df_train_final= dataFrame.df_train[df_column]
        # get the column which contains the data the user would like to use for data processing
        st.write(
            'Please select the column name that contains the data you would like to process, click enter to confirm')

        df_column = st.selectbox('select a column', df_train_orgin.columns)

        st.write('Please select the type of text processing you would like to used')
        process_tweets = st.checkbox('Clean Tweets')
        tweets = st.checkbox('Process tweets data')
        tokenize = st.checkbox('Remove auxilary')
        clean_NonAscii = st.checkbox('Remove Ascii code')

        if st.button('Process'):
            st.session_state.processed = True

            if st.session_state.processed:
                try:
                    message, st.session_state.df_train = confirm_checkbox(
                        process_tweets, tweets, tokenize, clean_NonAscii, st.session_state.df_train, df_column)
                    st.session_state.df_train_final = st.session_state.df_train['text_PP']
                    if message:
                        st.write('process is done')
                        st.write(st.session_state.df_train_final)
                        csv = convert_df(st.session_state.df_train_final)
                        csv2 = convert_df(st.session_state.df_train)

                        st.download_button(
                            label="Download processed data as CSV",
                            data=csv,
                            file_name='pre-processed_data.csv',
                            mime='text/csv',
                        )
                        st.download_button(
                            label="Download original and processed data as CSV",
                            data=csv2,
                            file_name='pre-processed_data_all.csv',
                            mime='text/csv',
                        )
                except:
                    st.warning(
                        'The selected column is unable to process, please re-select agian')
