import pandas as pd
import numpy as np
from deepface import DeepFace
import cv2


############################################################################################################
#                                     EMBEDDING
############################################################################################################

def embedding_image(img_array, model):
    """
    Convertit une image en vecteur d'embedding facial en utilisant DeepFace.

    Paramètres :
        img_array (np.ndarray) : Image chargée avec OpenCV (format BGR).
        model (str) : Nom du modèle DeepFace à utiliser (ex : "Facenet", "VGG-Face", etc.).

    Retour :
        pd.DataFrame : Embedding du visage sous forme de DataFrame (1 ligne, n colonnes), ou None si échec.
    """
    if img_array is not None:
        # Conversion en RGB car DeepFace attend ce format
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        embedding = represent_face_from_array(img_array, model)

        if embedding is not None:
            return pd.DataFrame(np.array(embedding[0]['embedding'])).T

    return None


def represent_face_from_array(img_array, model):
    """
    Extrait un vecteur d'embedding facial depuis une image RGB à l'aide de DeepFace.

    Paramètres :
        img_array (np.ndarray) : Image en RGB (format NumPy array).
        model (str) : Nom du modèle DeepFace à utiliser.

    Retour :
        list : Liste contenant un dictionnaire avec l'embedding et d'autres infos (si visage détecté).
    """
    embedding = DeepFace.represent(img_array, model_name=model, enforce_detection=False)
    return embedding


############################################################################################################
#                                     NORMALISATION
############################################################################################################

def delorean_normalisation(X):
    """
    Applique une normalisation L2 aux vecteurs (chaque ligne devient un vecteur unitaire).

    Paramètres :
        X (pd.DataFrame) : DataFrame contenant des vecteurs (chaque ligne = un vecteur).

    Retour :
        pd.DataFrame : DataFrame normalisée, où chaque ligne a une norme L2 de 1.
    """
    return X.div(X.pow(2).sum(axis=1)**0.5, axis=0)
