import os
import re
import numpy as np
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


############################################################################################################
#                              UTILS
############################################################################################################

def original_painting_title_back(filename):
    """
    Restaure le nom original d'un tableau à partir d'un nom de fichier avec suffixe "_face_...".

    Paramètre :
        filename (str) : Nom de fichier incluant un suffixe de type "_face_123_model_xyz.jpg".

    Retour :
        str : Nom du fichier d'origine avec l'extension conservée.
    """
    name, ext = os.path.splitext(filename)
    name = re.sub(r"_face_\d+.*", "", name)
    return name + ext


def find_file_in_subfolders(filename, root_folder):
    """
    Recherche récursivement un fichier dans tous les sous-dossiers d'un dossier racine.

    Paramètres :
        filename (str) : Nom du fichier à rechercher.
        root_folder (str) : Dossier racine dans lequel effectuer la recherche.

    Retour :
        str | None : Chemin complet du fichier trouvé, ou None si introuvable.
    """
    for folder_path, subfolders, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(folder_path, filename)
    return None


############################################################################################################
#                              COMPARISON METHODS
############################################################################################################

def cosine_model(X, y, neighbors, input_photo):
    """
    Recherche les voisins les plus similaires à une image via la similarité cosinus.

    Paramètres :
        X (pd.DataFrame) : Base d'embeddings indexée par les noms de fichiers des visages.
        y (np.ndarray) : Embedding de l'image cible (shape = [1, n]).
        neighbors (int) : Nombre de voisins à retourner.
        input_photo (str) : Chemin ou nom de la photo cible.

    Retour :
        dict : Résultat contenant le nom de l'image de départ, et les voisins similaires avec infos :
            - index (str) : nom du fichier "_face_..."
            - similarity (float)
            - painting_face (np.ndarray) : image visage
            - original_painting (np.ndarray) : image du tableau original
            - original_painting_artist (str)
            - original_painting_title (str)
            - original_painting_wikiart_link (str)
    """
    cosine_sim = cosine_similarity(y, X)
    n_neighbors = neighbors

    nearest_indices = np.argsort(-cosine_sim, axis=1)[:, :n_neighbors]

    results = {
        "input_photo": input_photo,
        "neighbors": []
    }

    for j in range(n_neighbors):
        neighbor_index = X.index[nearest_indices[0][j]]
        similarity = cosine_sim[0][nearest_indices[0][j]]

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        faces_dir = os.path.join(base_dir, "processed_data/5000visages_model_7_6")
        painting_face_path = os.path.join(faces_dir, neighbor_index)
        painting_face = mpimg.imread(painting_face_path)

        original_painting_name = original_painting_title_back(neighbor_index)
        root_folder = os.path.join(base_dir, "data/wikiart")
        original_painting_path = find_file_in_subfolders(original_painting_name, root_folder)
        original_painting = mpimg.imread(original_painting_path)

        original_painting_artist_raw, original_painting_title_raw = original_painting_name.split("_", 1)
        original_painting_artist = original_painting_artist_raw.replace("-", " ").title()
        title_no_ext = os.path.splitext(original_painting_title_raw)[0]
        original_painting_title = title_no_ext.replace("-", " ").title()
        original_painting_wikiart_link = f"https://www.wikiart.org/fr/{original_painting_artist_raw}/{title_no_ext}"

        results["neighbors"].append({
            "index": neighbor_index,
            "similarity": similarity,
            "painting_face": painting_face,
            "original_painting": original_painting,
            "original_painting_artist": original_painting_artist,
            "original_painting_title": original_painting_title,
            "original_painting_wikiart_link": original_painting_wikiart_link,
        })

    return results


def KNN_model(X, y, neighbors, input_photo):
    """
    TODO : Implémentation de la recherche des plus proches voisins via KNN.

    Paramètres :
        X (pd.DataFrame) : Base d'embeddings.
        y (np.ndarray) : Embedding de l'image cible.
        neighbors (int) : Nombre de voisins à retourner.
        input_photo (str) : Image cible.

    Retour :
        dict : Résultat structuré comme dans cosine_model.
    """
    # À implémenter si besoin
    return {
        "input_photo": input_photo,
        "neighbors": []
    }


############################################################################################################
#                                     COMPARISON
############################################################################################################

def compare(X, y, neighbors, comparison, input_photo):
    """
    Compare un embedding à une base en choisissant une méthode de similarité.

    Paramètres :
        X (pd.DataFrame) : Base d'embeddings indexée par noms de fichiers.
        y (np.ndarray) : Embedding à comparer (1 vecteur).
        neighbors (int) : Nombre de voisins à retourner.
        comparison (str) : Méthode de comparaison ('cosine' ou 'KNN').
        input_photo (str) : Image cible.

    Retour :
        dict : Résultats formatés par la méthode choisie.
    """
    print(f"Embedding model: on X name, comparison model: {comparison}")
    if comparison == "cosine":
        return cosine_model(X, y, neighbors, input_photo)
    if comparison == "KNN":
        return KNN_model(X, y, neighbors, input_photo)
