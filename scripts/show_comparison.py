import matplotlib.pyplot as plt

def show_comparison(result):
    """
    Affiche les résultats de comparaison sous forme de grille d'images.

    Chaque ligne correspond à un voisin similaire :
        - Colonne 1 : Image de départ (input)
        - Colonne 2 : Visage extrait du tableau similaire
        - Colonne 3 : Tableau original avec titre et artiste

    Paramètre :
        result (dict) : Dictionnaire retourné par la fonction de comparaison (cosine_model ou KNN_model).
                        Il contient une clé "input_photo" (image en tableau NumPy)
                        et une liste "neighbors" avec les informations des correspondances.
    """
    neighbors = result["neighbors"]
    n_neighbors = len(neighbors)

    # Une ligne par voisin, 3 colonnes : input, face, painting
    fig, axes = plt.subplots(n_neighbors, 3, figsize=(15, 5 * n_neighbors))

    # Si un seul voisin, axes est un tableau 1D → forcer à 2D pour simplifier le code
    if n_neighbors == 1:
        axes = [axes]

    for i, neighbor in enumerate(neighbors):
        # Colonne 1 : Visage d'entrée
        axes[i][0].imshow(result["input_photo"])
        axes[i][0].set_title("Input face")
        axes[i][0].axis('off')

        # Colonne 2 : Visage extrait du tableau
        if neighbor.get("painting_face") is not None:
            axes[i][1].imshow(neighbor["painting_face"])
            axes[i][1].set_title(f"Match {i+1} - Face")
        else:
            axes[i][1].set_title(f"Match {i+1} - Visage introuvable")
        axes[i][1].axis('off')

        # Colonne 3 : Tableau original
        if neighbor.get("original_painting") is not None:
            axes[i][2].imshow(neighbor["original_painting"])
            title = f"{neighbor.get('original_painting_artist', 'Artiste inconnu')} - {neighbor.get('original_painting_title', 'Titre inconnu')}"
            axes[i][2].set_title(title)
        else:
            axes[i][2].set_title("Tableau original introuvable")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()
