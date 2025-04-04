import streamlit as st
import pandas as pd

st. title('Serenas playground')

st.text('This is an app that I made for Serena.')

uploaded_file = st.file_uploader('Upload your file here')

if uploaded_file:
  df = pd.read_csv(uploaded_file)
  st.write(df.describe())
