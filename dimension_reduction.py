import numpy as np
import pandas as pd
import streamlit as st
import openml
import plotly.express as px
import plotly.graph_objects as go

import keras
from keras import layers
from keras import activations
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pca(feats, indices):
    model = PCA(n_components=3)
    results = model.fit_transform(feats[indices, :])
    
    return results


def kpca(feats, indices):
    kernel = st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'cosine'])
    
    model = KernelPCA(n_components=3, kernel=kernel)
    results = model.fit_transform(feats[indices, :])
    
    return results


def isomap(feats, indices):
    model = Isomap(n_components=3)
    results = model.fit_transform(feats[indices, :])
    
    return results


def tsne(feats, indices):
    perplexity = st.slider('Perplexity',
                           min_value=5,
                           max_value=50,
                           value=30,
                           step=1)
    
    model = TSNE(n_components=3, 
                 n_iter=500,
                 n_iter_without_progress=100,
                 early_exaggeration=20,
                 perplexity=perplexity, 
                 method='barnes_hut',
                 angle=1)
    results = model.fit_transform(feats[indices, :])
    
    return results


def umap(feats, indices):
    metric = st.selectbox('Metric', [
        'euclidean',
        'manhattan',
        'chebyshev',
        'minkowski',
        'canberra',
        'braycurtis',
        'mahalanobis',
        'wminkowski',
        'seuclidean',
        'cosine',
        'correlation'
    ])
    n_neighbors = st.slider('N Neighbors',
                            min_value=2,
                            max_value=200,
                            value=15,
                            step=1)
    min_dist = st.slider('Minimum Distance',
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.01)
        
    model = UMAP(n_components=3,
                 n_neighbors=n_neighbors,
                 min_dist=min_dist,
                 metric=metric)
    
    results = model.fit_transform(feats[indices, :])
    
    return results
    

def ae(feats, indices):
    hidden_size = st.slider('Hidden Size',
                            min_value=3,
                            max_value=feats.shape[1],
                            value=int(np.sqrt(feats.shape[1])),
                            step=1)
    n_epochs = st.slider('Number of Epochs',
                            min_value=1,
                            max_value=20,
                            value=5,
                            step=1)
    activation = st.selectbox('Activation', [
        None,
        'relu',
        'sigmoid',
        'tanh'
    ], index=1)
    
    feat_max = np.max(feats)
    
    inputs = keras.Input(shape=(784,))
    hidden1 = layers.Dense(hidden_size, activation=activation)(inputs)
    hidden1_bn1 = layers.BatchNormalization()(hidden1)
    encoded = layers.Dense(3)(hidden1_bn1)
    encoded_bn = layers.BatchNormalization()(encoded)
    hidden2 = layers.Dense(hidden_size, activation=activation)(encoded_bn)
    hidden2_bn = layers.BatchNormalization()(hidden2)
    decoded = layers.Dense(784, activation='sigmoid')(hidden2_bn)
    autoencoder = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded_bn)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(feats / feat_max, feats / feat_max,
                    epochs=n_epochs,
                    batch_size=256,
                    shuffle=True)
    
    results = encoder.predict(feats[indices, :] / feat_max)
   
    return results
