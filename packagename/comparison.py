import os
import re
import numpy as np
import pandas as pd
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

import re

def extract_face_number(filename):
    """
    Extrait le nombre après 'face_' dans un nom de fichier, que ce soit suivi d'un underscore, d'un point ou de la fin du nom.

    Paramètre :
        filename (str) : Nom du fichier contenant un segment de type 'face_123', 'face_123_', ou 'face_123.jpg'.

    Retour :
        int ou None : Le nombre extrait après 'face_', ou None s'il n'est pas trouvé.
    """
    match = re.search(r"face_(\d+)(?:_|\.|$)", filename)
    if match:
        return int(match.group(1))
    return None



def wikiart_url_to_image_url(wikiart_link: str) -> str:
    # Enlever la langue (ex: /fr/)
    url_parts = wikiart_link.replace("https://www.wikiart.org/", "").split("/")
    if len(url_parts) < 2:
        return None

    # Extraire l’artiste et le titre
    artist_raw = url_parts[-2]
    title_raw = url_parts[-1]

    # Concaténer dans l’URL finale
    return f"https://uploads3.wikiart.org/images/{artist_raw}/{title_raw}.jpg!Large.jpg"


import pandas as pd

def get_face_coordinates_from_csv(csv_path, target_filename):
    """
    Lit un CSV avec pandas et retourne les coordonnées et la taille de l'image
    associées à un nom de fichier donné.

    :param csv_path: Chemin vers le fichier CSV
    :param target_filename: Nom de fichier exact à chercher (ex: "mon_image_face_1.jpg")
    :return: Liste de tuples (x1, y1, x2, y2, width, height) ou [] si rien trouvé
    """
    try:
        df = pd.read_csv(csv_path)

        # Filtrage
        filtered = df[df['filename'] == target_filename]

        # Vérifie que toutes les colonnes nécessaires sont bien là
        required_cols = {'x1', 'y1', 'x2', 'y2', 'orig_width', 'orig_height'}
        print( list(filtered[['x1', 'y1', 'x2', 'y2', 'orig_width', 'orig_height']].itertuples(index=False, name=None))
)
        if not required_cols.issubset(set(df.columns)):
            print("❌ Colonnes manquantes dans le CSV.")
            return []

        # Extraction sous forme de tuples
        return list(filtered[['x1', 'y1', 'x2', 'y2', 'orig_width', 'orig_height']].itertuples(index=False, name=None))

    except FileNotFoundError:
        print(f"❌ Fichier CSV non trouvé : {csv_path}")
    except Exception as e:
        print(f"⚠️ Erreur pendant la lecture du CSV : {e}")

    return []




def get_image_url_from_author_title(neighbor_index, csv_path):
    # Retire le suffixe du fichier (ex: "_398.jpg" ou ".jpeg")
    if neighbor_index.endswith(".jpeg") or neighbor_index.endswith(".jpg"):
        base_name = "_".join(neighbor_index.split("_")[:-1])
    else:
        base_name = neighbor_index

    # Chargement du CSV
    df = pd.read_csv(csv_path)

    # Vérifie si le nom de base est présent dans la colonne 'author_title'
    if base_name in df['author_title'].values:
        image_url = df.loc[df['author_title'] == base_name, 'image_url'].values[0]
        return image_url

    return False




############################################################################################################
#                              COMPARISON METHODS
############################################################################################################
def cosine_model(X, y, neighbors):
    """
    Recherche les voisins les plus similaires à une image via la similarité cosinus.
    """
    # Calcul de la similarité
    cosine_sim = cosine_similarity(y, X)

    # Récupération des indices des k plus proches voisins
    nearest_indices = np.argsort(-cosine_sim, axis=1)[:, :neighbors]

    results = {
        "neighbors": []
    }

    for j in range(neighbors):
        try:
            neighbor_index = X.index[nearest_indices[0][j]]
            similarity = cosine_sim[0][nearest_indices[0][j]]

            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
            original_painting_name = original_painting_title_back(neighbor_index)

            if "_" not in original_painting_name:
                print(f"❌ Nom de fichier invalide (pas de _): {original_painting_name}")
                continue

            original_painting_artist_raw, original_painting_title_raw = original_painting_name.split("_", 1)
            original_painting_artist = original_painting_artist_raw.replace("-", " ").title()
            title_no_ext = os.path.splitext(original_painting_title_raw)[0]
            original_painting_title = title_no_ext.replace("-", " ").title()
            face_index = extract_face_number(neighbor_index)

            # Chemin vers les données CSV
            csv_path_coordinates = os.path.join(base_dir, "../sources/faces_coordinates.csv")
            face_coordinates = get_face_coordinates_from_csv(csv_path_coordinates, neighbor_index)
            csv_path_ruth = os.path.join(base_dir, "../sources/url_additionnal_paintings.csv")

            # Image URL
            image_url = get_image_url_from_author_title(neighbor_index, csv_path_ruth)
            if image_url is False:
                original_painting_wikiart_link = f"https://www.wikiart.org/fr/{original_painting_artist_raw}/{title_no_ext}"
                original_painting_image_url = wikiart_url_to_image_url(original_painting_wikiart_link)
            else:
                original_painting_wikiart_link = None
                original_painting_image_url = image_url


            results["neighbors"].append({
                "index": neighbor_index,
                "face_index": face_index,
                "face_coordinates": face_coordinates,
                "similarity": round(similarity, 3),
                "original_painting_artist": original_painting_artist,
                "original_painting_title": original_painting_title,
                "original_painting_wikiart_link": original_painting_wikiart_link,
                "original_painting_image_url": original_painting_image_url,
                            })

        except Exception as e:
            print(f"❌ Erreur lors du traitement du voisin {j+1} : {e}")
            continue

    return results




def KNN_model(X,y,neighbors,algorithm='auto',leaf_size=30,metric='minkowski'):
    """
    Implémentation de la recherche des plus proches voisins via KNN.

    Paramètres :
        X (pd.DataFrame) : Base d'embeddings.
        y (np.ndarray) : Embedding de l'image cible.
        neighbors (int) : Nombre de voisins à retourner.

    Retour :
        dict : Résultat structuré comme dans cosine_model.
    """
    n_neighbors = 3
    knn = NearestNeighbors(
    n_neighbors=neighbors,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    n_jobs=-1)

    knn.fit(X)

    distances, indices = knn.kneighbors(y)

    results = {
        "neighbors": []
    }

    for j in range(n_neighbors):
        neighbor_index = X.index[indices[0][j]]
        similarity = distances[j]

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        original_painting_name = original_painting_title_back(neighbor_index)

        if "_" not in original_painting_name:
            print(f"❌ Nom de fichier invalide (pas de _): {original_painting_name}")
            continue

        original_painting_artist_raw, original_painting_title_raw = original_painting_name.split("_", 1)
        original_painting_artist = original_painting_artist_raw.replace("-", " ").title()
        title_no_ext = os.path.splitext(original_painting_title_raw)[0]
        original_painting_title = title_no_ext.replace("-", " ").title()
        face_index = extract_face_number(neighbor_index)

        # Chemin vers les données CSV
        csv_path_coordinates = os.path.join(base_dir, "../sources/faces_coordinates.csv")
        face_coordinates = get_face_coordinates_from_csv(csv_path_coordinates, neighbor_index)
        csv_path_ruth = os.path.join(base_dir, "../sources/url_additionnal_paintings.csv")

        # Image URL
        image_url = get_image_url_from_author_title(neighbor_index, csv_path_ruth)
        if image_url is False:
            original_painting_wikiart_link = f"https://www.wikiart.org/fr/{original_painting_artist_raw}/{title_no_ext}"
            original_painting_image_url = wikiart_url_to_image_url(original_painting_wikiart_link)
        else:
            original_painting_wikiart_link = None
            original_painting_image_url = image_url

        print(f"❌ Similarity {similarity}")

        results["neighbors"].append({
                "index": neighbor_index,
                "face_index": face_index,
                "face_coordinates": face_coordinates,
                "similarity": round(similarity, 3),
                "original_painting_artist": original_painting_artist,
                "original_painting_title": original_painting_title,
                "original_painting_wikiart_link": original_painting_wikiart_link,
                "original_painting_image_url": original_painting_image_url,
                            })

    return results


############################################################################################################
#                                     COMPARISON
############################################################################################################

def compare(X, y, neighbors, comparison):
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
        return cosine_model(X, y, neighbors)
    if comparison == "KNN":
        return KNN_model(X, y, neighbors)
