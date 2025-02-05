import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def highlight_pivot(val, is_pivot):
    """Met en surbrillance la cellule du pivot."""
    color = 'background-color: yellow' if is_pivot else ''
    return color

def methode_simplexe(c, A, b):
    num_variables = len(c)
    num_contraintes = len(b)
    c = -c  # Transformation en minimisation

    tableau = np.hstack((A, np.eye(num_contraintes), b.reshape(-1, 1))).astype(float)
    ligne_c = np.hstack((c, np.zeros(num_contraintes + 1)))
    tableau = np.vstack([tableau, ligne_c])
    steps = [("Forme Standard", pd.DataFrame(tableau))]

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
                tableau[i, :] -= tableau[i, colonne_pivot] * tableau[ligne_pivot, :]
        
        step_name = f"Itération {len(steps)}"
        steps.append((step_name, pd.DataFrame(tableau)))
    
    solution = np.zeros(num_variables)
    for j in range(num_variables):
        colonne = tableau[:-1, j]
        if np.sum(colonne == 1) == 1 and np.sum(colonne == 0) == num_contraintes - 1:
            ligne = np.where(colonne == 1)[0][0]
            solution[j] = tableau[ligne, -1]
    
    valeur_optimale = -tableau[-1, -1]
    return solution, valeur_optimale, steps

st.title("Méthode du Simplexe")

c_input = st.text_input("Fonction objectif (séparée par des virgules)", "4470, 2310, 2650")
c = np.array([float(x) for x in c_input.split(",")])

A = []
b = []
for i in range(3):
    st.write(f"Contrainte {i+1}:")
    A_input = st.text_input(f"Coefficients Contrainte {i+1}", "1, 1, 1" if i == 0 else "6400, 6400, 7200" if i == 1 else "900, 600, 4500")
    b_input = st.text_input(f"Second membre Contrainte {i+1}", "1000" if i == 0 else "7000000" if i == 1 else "1600000")
    A.append([float(x) for x in A_input.split(",")])
    b.append(float(b_input))

if st.button("Résoudre"):
    A = np.array(A)
    b = np.array(b)
    solution, valeur_optimale, steps = methode_simplexe(c, A, b)
    st.session_state.steps = steps
    st.session_state.solution = solution
    st.session_state.valeur_optimale = valeur_optimale
    
    st.header("Résultats")
    for i, val in enumerate(solution):
        st.write(f"X{i+1} = {round(val, 2)}")
    st.write(f"**Valeur optimale :** {round(valeur_optimale, 2)}")

if 'steps' in st.session_state:
    if st.button("Afficher toutes les étapes"):
        st.header("Étapes du Simplexe")
        for i, (step_name, step) in enumerate(st.session_state.steps):
            st.write(f"**{step_name}**")
            styled_table = step.style.applymap(lambda val: highlight_pivot(val, False))
            st.dataframe(styled_table)
