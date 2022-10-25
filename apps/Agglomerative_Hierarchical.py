
import streamlit as st
import PyQt5
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import matplotlib
import plotly.figure_factory as ff
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
matplotlib.use("agg")
import base64
from PIL import Image

def app():

    AgglomerativeClustering().get_params()

 
    X = st.session_state.df_train['text_PP']

    # Construct a pipeline: TF-IDF -> Sparse to Dense -> Clustering
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english'),
        FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
        AgglomerativeClustering(linkage='average')  # Use average linkage
    )

    pipeline = pipeline.fit(X)
    pipeline.named_steps
    model = pipeline.named_steps['agglomerativeclustering']




    def plot_dendrogram(model, **kwargs):
        # Children of hierarchical clustering
        children = model.children_

        # Distances between each pair of children
        # Since we don't have this information, we can use a uniform one for plotting
        distance = np.arange(children.shape[0])

        # The number of observations contained in each cluster level
        no_of_observations = np.arange(2, children.shape[0] + 2)

        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    # def render_svg(svg):
    #     """Renders the given svg string."""
    #     b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    #     html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    #     st.write(html, unsafe_allow_html=True)


    plot_dendrogram(model, labels=X.index, orientation='right',p=10,truncate_mode='level')
    # with _lock:
    #     st.pyplot(plot_dendrogram)
    plt.savefig('./output/AH.jpg')

    image = Image.open('./output/AH.jpg')
    st.image(image)


    # fig = ff.create_dendrogram(X, orientation='left', labels=X.index)
    # fig.update_layout(width=800, height=800)
    # fig.show()

    #st.pyplot(plot_dendrogram)