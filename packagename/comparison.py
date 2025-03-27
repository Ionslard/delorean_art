import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from bs4 import BeautifulSoup

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
    name= re.sub(r"\(\d+\)$", "", name).strip()
    return name + ext


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



def wikiart_url_to_image_url(wikiart_link):
    """
    Extrait dynamiquement l'URL de l'image principale depuis une page WikiArt.

    Paramètre :
        wikiart_link (str) : Lien vers la page WikiArt du tableau.

    Retour :
        str : URL de l'image trouvée, ou None si échec.
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(wikiart_link, headers=headers)
        if response.status_code != 200:
            print(f"❌ Erreur HTTP {response.status_code} pour {wikiart_link}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        og_image = soup.find("meta", property="og:image")

        if og_image and og_image.get("content"):
            return og_image["content"]
        else:
            print(f"⚠️ Aucun lien og:image trouvé sur {wikiart_link}")
            return None

    except Exception as e:
        print(f"⚠️ Erreur lors de l'accès à la page WikiArt : {e}")
        return None


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
        #print( list(filtered[['x1', 'y1', 'x2', 'y2', 'orig_width', 'orig_height']].itertuples(index=False, name=None)))
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
    # Enlève l'extension d'abord
    base = re.sub(r'\.jpe?g$', '', neighbor_index)
    # Supprime la partie "_xxx" à la fin (où xxx = chiffres ou "face1" par ex)_
    base_name = re.sub(r'(_\d+.*)$', '', base)
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
def KNN_model_ar(X,y,neighbors,algorithm='auto',leaf_size=30,metric='minkowski'):
    """
    Implémentation de la recherche des plus proches voisins via KNN.

    Paramètres :
        X (pd.DataFrame) : Base d'embeddings.
        y (np.ndarray) : Embedding de l'image cible.
        neighbors (int) : Nombre de voisins à retourner.

    Retour :
        dict : Résultat structuré comme dans cosine_model.
    """
    n_neighbors = neighbors
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
        similarity = distances[0][j]

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
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
        csv_path_coordinates = os.path.join(base_dir,"csv_source/faces_coordinates.csv")
        face_coordinates = get_face_coordinates_from_csv(csv_path_coordinates, neighbor_index)
        csv_path_ruth = os.path.join(base_dir,"csv_source/url_additionnal_paintings.csv")

        # Image URL
        image_url = get_image_url_from_author_title(neighbor_index, csv_path_ruth)
        if image_url is False:#pas dans les additionnal paintings donc from wikiart
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

    return results

def KNN_model(X, y, neighbors, algorithm='auto', leaf_size=30, metric='minkowski'):
    """
    Implémentation de la recherche des plus proches voisins via KNN.

    Paramètres :
        X (pd.DataFrame) : Base d'embeddings.
        y (np.ndarray) : Embedding de l'image cible.
        neighbors (int) : Nombre de voisins valides à retourner.

    Retour :
        dict : Résultat structuré comme dans cosine_model.
    """
    knn = NearestNeighbors(
        n_neighbors=min(50, len(X)),  # on prend un grand nombre de voisins (ou tous)
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        n_jobs=-1
    )

    knn.fit(X)
    distances, indices = knn.kneighbors(y)

    results = {
        "neighbors": []
    }

    collected = 0
    i = 0
    max_try = len(indices[0])

    while collected < neighbors and i < max_try:
        neighbor_index = X.index[indices[0][i]]
        similarity = distances[0][i]
        i += 1

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        original_painting_name = original_painting_title_back(neighbor_index)

        if "_" not in original_painting_name:
            print(f"❌ Nom de fichier invalide (pas de _): {original_painting_name}")
            continue

        original_painting_artist_raw, original_painting_title_raw = original_painting_name.split("_", 1)
        original_painting_artist = original_painting_artist_raw.replace("-", " ").title()
        title_no_ext = os.path.splitext(original_painting_title_raw)[0]
        original_painting_title = title_no_ext.replace("-", " ").title()
        face_index = extract_face_number(neighbor_index)

        csv_path_coordinates = os.path.join(base_dir, "csv_source/faces_coordinates.csv")
        face_coordinates = get_face_coordinates_from_csv(csv_path_coordinates, neighbor_index)
        csv_path_ruth = os.path.join(base_dir, "csv_source/url_additionnal_paintings.csv")

        image_url = get_image_url_from_author_title(neighbor_index, csv_path_ruth)

        if image_url is False:
            original_painting_wikiart_link = f"https://www.wikiart.org/fr/{original_painting_artist_raw}/{title_no_ext}"

            # try:
            #     response = requests.get(original_painting_wikiart_link, timeout=3)
            #     if response.status_code != 200:
            #         print(f"⚠️ Lien WikiArt invalide : {original_painting_wikiart_link}")
            #         continue
            # except requests.RequestException:
            #     print(f"⚠️ Erreur lors de l'accès à : {original_painting_wikiart_link}")
            #     continue

            original_painting_image_url = wikiart_url_to_image_url(original_painting_wikiart_link)
        else:
            original_painting_wikiart_link = None
            original_painting_image_url = image_url
        if not face_coordinates:  # Si vide ou None
            print(f"⏭️ Visage ignoré car pas de coordonnées : {neighbor_index}")
            continue
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

        collected += 1

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
   # print(f"Embedding model: on X name, comparison model: {comparison}")
    if comparison == "cosine":
        return cosine_model(X, y, neighbors)
    if comparison == "KNN":
        return KNN_model(X, y, neighbors)
