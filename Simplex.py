import streamlit as st
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

def afficher_tableau(tableau, iteration, ligne_pivot=None, colonne_pivot=None):
    st.write(f"**Tableau Simplexe ({'Forme Standard' if iteration == 0 else f'Itération {iteration}'}) :**")
    df = pd.DataFrame(tableau)
    
    if ligne_pivot is not None and colonne_pivot is not None:
        def highlight_pivot(s, lp=ligne_pivot, cp=colonne_pivot):
            return [
                "background-color: yellow" if s.name == lp or i == cp else ""
                for i in range(len(s))
            ]
        
        styled_df = df.style.apply(highlight_pivot, axis=1)
        styled_df.set_properties(subset=pd.IndexSlice[ligne_pivot, colonne_pivot], **{'border': '3px solid red'})
        st.dataframe(styled_df)
        st.write(f"\u2192 Le pivot est en position ({ligne_pivot}, {colonne_pivot})")
    else:
        st.dataframe(df)

def methode_simplexe(c, A, b):
    num_variables = len(c)
    num_contraintes = len(b)
    tableau = np.hstack((A, np.eye(num_contraintes), b.reshape(-1, 1))).astype(float)
    ligne_c = np.hstack((c, np.zeros(num_contraintes + 1)))
    tableau = np.vstack([tableau, ligne_c])
    
    afficher_tableau(tableau, 0)
    iteration = 1
    steps = []
    while True:
        colonne_pivot = np.argmin(tableau[-1, :-1])
        if tableau[-1, colonne_pivot] >= 0:
            break
        ratios = tableau[:-1, -1] / tableau[:-1, colonne_pivot]
        ratios[ratios <= 0] = np.inf
        ligne_pivot = np.argmin(ratios)
        
        pivot = tableau[ligne_pivot, colonne_pivot]
        tableau[ligne_pivot, :] /= pivot
        
        for i in range(len(tableau)):
            if i != ligne_pivot:
                facteur = tableau[i, colonne_pivot]
                tableau[i, :] -= facteur * tableau[ligne_pivot, :]
        
        steps.append(pd.DataFrame(tableau))
        afficher_tableau(tableau, iteration, ligne_pivot, colonne_pivot)
        iteration += 1
    
    solution = np.zeros(num_variables)
    for i in range(num_contraintes):
        base_var = np.where(tableau[i, :num_variables] == 1)[0]
        if len(base_var) == 1:
            solution[base_var[0]] = tableau[i, -1]
    
    valeur_optimale = tableau[-1, -1]
    return solution, -valeur_optimale, steps

st.title("Méthode du Simplexe")
st.header("Données du problème")
st.subheader("Fonction objectif :")
c_input = st.text_input("Coefficients (séparés par des virgules) :", "-4470, -2310, -2650")
c = np.array([float(x) for x in c_input.split(",")])
st.subheader("Contraintes :")
st.write("Contrainte 1 :")
A1_input = st.text_input("Coefficients :", "1, 1, 1")
b1_input = st.text_input("Second membre :", "1000")
A1 = np.array([float(x) for x in A1_input.split(",")])
b1 = float(b1_input)
st.write("Contrainte 2 :")
A2_input = st.text_input("Coefficients :", "6400, 6400, 7200")
b2_input = st.text_input("Second membre :", "7000000")
A2 = np.array([float(x) for x in A2_input.split(",")])
b2 = float(b2_input)
st.write("Contrainte 3 :")
A3_input = st.text_input("Coefficients :", "900, 600, 4500")
b3_input = st.text_input("Second membre :", "1600000")
A3 = np.array([float(x) for x in A3_input.split(",")])
b3 = float(b3_input)
if st.button("Résoudre"):
    A = np.array([A1, A2, A3])
    b = np.array([b1, b2, b3])
    solution, valeur_optimale, steps = methode_simplexe(c, A, b)
    st.header("Résultats :")
    for i in range(len(solution)):
        st.write(f"X{i + 1} : {round(solution[i], 2)}")
    st.write(f"Valeur optimale : {round(valeur_optimale, 2)}")
    if st.button("Afficher toutes les étapes"):
        st.header("Étapes du Simplexe :")
        for i, step in enumerate(steps):
            st.write(f"**Itération {i + 1}**")
            st.dataframe(step)
