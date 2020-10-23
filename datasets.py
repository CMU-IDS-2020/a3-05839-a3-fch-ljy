import streamlit as st
import openml
import pandas as pd

@st.cache
def mnist():
    mnist = openml.datasets.get_dataset('mnist_784')
    x, y, categorical, attribute_names = mnist.get_data()
    feats = x[:2000].drop('class', axis='columns').to_numpy().astype('float32')
    labels = x['class'].to_numpy()
    raw = x
    return feats, labels, raw

@st.cache
def mnist_csv():
    x = pd.read_csv('mnist.csv')
    feats = x.drop('class', axis='columns').to_numpy().astype('float32')
    labels = x['class'].astype(str).to_numpy()
    raw = x
    return feats, labels, raw