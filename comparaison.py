import matplotlib.pyplot as plt

# Données
methodes = ["Jacobi", "Gauss-Seidel", "Décomposition LU", "Pivot de Gauss"]
precisions = [64.93, 41.55, 28.57, 37.39]

# Couleurs
couleurs = ['blue', 'green', 'orange', 'red']

# Création du graphique
plt.figure(figsize=(10, 6))
plt.barh(methodes, precisions, color=couleurs)
plt.xlabel('Précision (%)')
plt.ylabel('Méthode')
plt.title('Comparaison de la précision des méthodes')
plt.gca().invert_yaxis()  # Inverser l'axe des Y pour que la méthode la plus précise soit en haut

# Affichage du graphique
plt.show()