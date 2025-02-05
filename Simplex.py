import streamlit as st
import numpy as np
import pandas as pd

# Fonction de la méthode du simplexe (nouvelle implémentation)
def methode_simplexe(c, A, b):
    num_variables = len(c)
    num_contraintes = len(b)

    # Transformation en minimisation
    c = -c  

    # Création du tableau du simplexe
    tableau = np.hstack((A, np.eye(num_contraintes), b.reshape(-1, 1))).astype(float)

    # Ajout de la ligne de la fonction objectif
    ligne_c = np.hstack((c, np.zeros(num_contraintes + 1)))
    tableau = np.vstack([tableau, ligne_c])

    iteration = 0
    steps = [pd.DataFrame(tableau)]

    while True:
        # Sélection de la colonne pivot (variable entrante)
        colonne_pivot = np.argmin(tableau[-1, :-1])  

        # Vérification de l'optimalité
        if tableau[-1, colonne_pivot] >= 0:
            break  

        # Sélection de la ligne pivot (variable sortante)
        ratios = tableau[:-1, -1] / tableau[:-1, colonne_pivot]
        ratios[ratios <= 0] = np.inf  
        ligne_pivot = np.argmin(ratios)

        # Normalisation de la ligne pivot
        pivot = tableau[ligne_pivot, colonne_pivot]
        tableau[ligne_pivot, :] /= pivot

        # Mise à zéro des autres lignes
        for i in range(len(tableau)):
            if i != ligne_pivot:
                tableau[i, :] -= tableau[i, colonne_pivot] * tableau[ligne_pivot, :]

        iteration += 1
        steps.append(pd.DataFrame(tableau))

    # Extraction des résultats
    solution = np.zeros(num_variables)
    for j in range(num_variables):
        colonne = tableau[:-1, j]  
        if np.sum(colonne == 1) == 1 and np.sum(colonne == 0) == num_contraintes - 1:
            ligne = np.where(colonne == 1)[0][0]
            solution[j] = tableau[ligne, -1]

    # Récupération de la valeur optimale
    valeur_optimale = -tableau[-1, -1]

    return solution, valeur_optimale, steps

# Interface Streamlit
st.title("Méthode du Simplexe")

# Entrée des données
st.header("Données du problème")

# Fonction objectif
st.subheader("Fonction objectif :")
c_input = st.text_input("Coefficients (séparés par des virgules) :", "4470, 2310, 2650")
c = np.array([float(x) for x in c_input.split(",")])

# Contraintes
st.subheader("Contraintes :")

# Contrainte 1
st.write("Contrainte 1 :")
A1_input = st.text_input("Coefficients :", "1, 1, 1")
b1_input = st.text_input("Second membre :", "1000")
A1 = np.array([float(x) for x in A1_input.split(",")])
b1 = float(b1_input)

# Contrainte 2
st.write("Contrainte 2 :")
A2_input = st.text_input("Coefficients :", "6400, 6400, 7200")
b2_input = st.text_input("Second membre :", "7000000")
A2 = np.array([float(x) for x in A2_input.split(",")])
b2 = float(b2_input)

# Contrainte 3
st.write("Contrainte 3 :")
A3_input = st.text_input("Coefficients :", "900, 600, 4500")
b3_input = st.text_input("Second membre :", "1600000")
A3 = np.array([float(x) for x in A3_input.split(",")])
b3 = float(b3_input)

# Variables pour le stockage des résultats
solution = None
valeur_optimale = None
steps = []

# Bouton pour résoudre
if st.button("Résoudre le problème"):
    # Création de la matrice A et du vecteur b
    A = np.array([A1, A2, A3])
    b = np.array([b1, b2, b3])

    # Résolution
    solution, valeur_optimale, steps = methode_simplexe(c, A, b)

    # Affichage des résultats
    st.header("Résultats :")
    for i in range(len(solution)):
        st.write(f"X{i+1} : {round(solution[i], 2)}")

    st.write(f"**Valeur optimale (Maximum de la fonction objectif) :** {round(valeur_optimale, 2)}")

# Bouton pour afficher toutes les étapes
if solution is not None and st.button("Afficher toutes les étapes"):
    st.header("Étapes du Simplexe :")
    for i, step in enumerate(steps):
        if i == 0:
            st.write('Forme Standard')
        st.write(f"**Itération {i}**")
        st.dataframe(step)
