# Import the required packages

import streamlit as st
import pandas as pd
import altair as alt
#import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# -------------------------
# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('Iris Classification')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("À propos", use_container_width=True, on_click=set_page_selection, args=('about',)):
	    st.session_state.page_selection = 'about'
	    st.write("""
     ## À propos de cette application
     
     Cette application est conçue pour **explorer** et **comprendre** le célèbre jeu de données des **fleurs d'Iris**. 
     
     **Qu'est-ce que le jeu de données Iris ?**
     
     C'est une base de données utilisée comme référence dans le domaine de l'apprentissage automatique. Elle contient des informations sur trois espèces d'Iris (Setosa, Versicolor et Virginica), telles que la longueur et la largeur de leurs sépales et pétales.
     
     **Que fait cette application ?**
     
     * **Présentation des données:** Visualisation des caractéristiques des fleurs d'Iris pour une meilleure compréhension.
     
     * **Prédiction:** Utilisation de modèles d'apprentissage automatique pour prédire l'espèce d'une fleur en fonction de ses mesures.
     
     **Objectif:**
     
     Notre objectif est de rendre l'apprentissage automatique et l'analyse de données plus accessibles, en utilisant un exemple concret et bien connu.
     """)
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------

# Load data
df = pd.read_csv('iris.csv', delimiter=',')

# Set page title
st.title('ISJM BI - Exploration des données des Iris')

st.header('Pré-analyse visuelles données données des Iris TP1')  # On définit l'en-tête d'une section


# Afficher les premières lignes des données chargées data
#st.write(df.head())
	
st.subheader('Description des données')  # Sets a subheader for a subsection

# Show Dataset
if st.checkbox("Boutons de prévisualisation du DataFrame"):
	if st.button("Head"):
		st.write(df.head(2))
	if st.button("Tail"):
		st.write(df.tail())
	if st.button("Infos"):
		st.write(df.info())
	if st.button("Shape"):
		st.write(df.shape)
	else:
		st.write(df.head(2))


# Create chart
chart = alt.Chart(df).mark_point().encode(
    x='petal_length',
    y='petal_width',
    color="species"
)

# Display chart
st.write(chart)

#Interactive design representation 
chart2 = alt.Chart(df).mark_circle(size=60).encode(
    x='sepal_length',
    y='sepal_width',
    color='species',
    tooltip=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
).interactive()

st.write(chart2)


# About

if st.button("About App"):
	st.subheader("App d'exploration des données des Iris")
	st.text("Contruite avec Streamlit")
	st.text("Thanks to the Streamlit Team Amazing Work")

if st.checkbox("By"):
	st.text("Stéphane C. K. Tékouabou")
	st.text("ctekouaboukoumetio@gmail.com")
