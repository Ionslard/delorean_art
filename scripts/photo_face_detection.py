import os
import cv2
from ultralytics import YOLO

############################################################################################################
#                                    FACE DETECTION
############################################################################################################

def delorean_photo_face_detection(photo):
    """
    Détecte un visage dans une image à l'aide du modèle YOLOv8 pré-entraîné (ultralytics).

    Paramètre :
        photo (np.ndarray) : Image déjà chargée (via OpenCV, format NumPy array, en BGR ou RGB).

    Retour :
        np.ndarray | None : Image du visage détecté (crop), ou None si aucun visage détecté.

    Remarques :
        - Utilise le modèle YOLOv8n personnalisé pour la détection faciale (yolov8n-face.pt).
        - Ne détecte qu'un seul visage (le premier trouvé).
    """
    if photo is None:
        print("❌ Erreur : aucune image chargée.")
        return None

    print("🔹 Détection du visage en cours...")

    # Chargement du modèle YOLO pour les visages
    yolo_model = YOLO("models/yolov8n-face.pt")
    results = yolo_model(photo)

    # Extraction de la première bounding box détectée
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = photo[y1:y2, x1:x2]
            if face_crop.size == 0:
                print("❌ Visage détecté mais zone vide.")
                return None
            print("✅ Visage détecté. Démarrage de la Delorean.")
            return face_crop

    print("❌ Aucun visage détecté.")
    return None
