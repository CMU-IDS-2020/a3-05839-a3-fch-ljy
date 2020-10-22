import numpy as np
import pandas as pd
import streamlit as st
import openml
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


st.title('Dimentionality Reduction')


def mnist():
    mnist = openml.datasets.get_dataset('mnist_784')
    x, y, categorical, attribute_names = mnist.get_data()
    feats = x.drop('class', axis='columns').to_numpy().astype('float32')
    labels = x['class'].to_numpy()
    return feats, labels


def pca(feats, n_samples):
    model = PCA(n_components=3).fit(feats)
    indices = np.random.choice(len(feats), n_samples, replace=False)
    results = model.transform(feats[indices, :])
    
    return results, indices


def kpca(feats, n_samples):
    kernel = st.selectbox('Kernel', ['linear', 'poly', 'rbf', 'cosine'])
    
    model = KernelPCA(n_components=3, kernel=kernel)
    indices = np.random.choice(len(feats), n_samples, replace=False)
    results = model.fit_transform(feats[indices, :])
    
    return results, indices


def tsne(feats, n_samples):
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
    indices = np.random.choice(len(feats), n_samples, replace=False)
    results = model.fit_transform(feats[indices, :])
    
    return results, indices


def umap(feats, n_samples):
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
                            value=0.1,
                            step=0.01)
        
    model = UMAP(n_components=3,
                 n_neighbors=n_neighbors,
                 min_dist=min_dist,
                 metric=metric)
    
    indices = np.random.choice(len(feats), n_samples, replace=False)
    results = model.fit_transform(feats[indices, :])
    
    return results, indices
    

def vae(feats, n_samples):
    pass
    
    
datasets = {'MNIST': mnist}
algorithms = {'PCA': pca,
              'KPCA': kpca,
              't-SNE': tsne,
              'UMAP': umap}

ds_opt = st.selectbox('Select a dataset:', list(datasets.keys()))
algo_opt = st.selectbox('Select an algorithm:', list(algorithms.keys()))


feats, labels = datasets[ds_opt]()

n_samples = st.slider('Number of Samples', 
                      min_value=500, 
                      max_value=len(feats), 
                      value=min(2500, len(feats)), 
                      step=500)

results, indices = algorithms[algo_opt](feats, n_samples)

reduced = pd.DataFrame(results, columns=['x', 'y', 'z'])
reduced['class'] = labels[indices]

fig = px.scatter_3d(reduced, x='x', y='y', z='z', color='class', opacity=1)
fig.update_layout(autosize=False,
                  width=700,
                  height=800)

st.plotly_chart(fig)