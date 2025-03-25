# TODO: Import your package, replace this by explicit imports of what you need
from packagename.photo_face_detection import delorean_photo_face_detection_coordinates
from packagename.photo_face_embedding import embedding_image, delorean_normalisation
from packagename.comparison import compare

########################################
# pip install opencv-python
# pip install python-multipart
########################################


# from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ajout de package
from fastapi import FastAPI, UploadFile, File, Response
import numpy as np
import cv2
import os
import pandas as pd
# from scripts.photo_face_detection import delorean_photo_face_detection_coordinates
# from scripts.photo_face_embedding import embedding_image, delorean_normalisation
# from scripts.comparison import compare
from fastapi.responses import JSONResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint
@app.get("/")
def root():
    return {
        'message': "The Delorean API is now live!"
    }

# comment Ruth: Endpoint pour charger une image
@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    try:
        ### Receiving the image
        contents = await img.read()
        ### Convertir en tableau NumPy
        nparr = np.fromstring(contents, np.uint8)
        ### Décoder l'image avec OpenCV
        cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
        ### Vérifier si l'image est bien décodée
        if cv2_img is None:
            return Response(content="Invalid image", media_type="text/plain", status_code=400)


        ### Faire le lien avec les modèles pour


        # 1. détecter les visages,
        input=delorean_photo_face_detection_coordinates(cv2_img)
        face_crop = input[0]
        if face_crop is None:
            return Response(content="No face detected", media_type="text/plain", status_code=404)

       # 2. embedder la partie croppée,
        embedding = embedding_image(face_crop, model="Facenet512")
        if embedding is None:
            return Response(content="Failed to compute embedding", media_type="text/plain", status_code=500)

        # Normalisation
        embedding = delorean_normalisation(embedding)

        # Chargement des embeddings
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        embedding_path = os.path.join(base_dir, "../csv_source/X_full_512_final.parquet")
        X=pd.read_parquet(embedding_path)

        # 3. comparer la matrice
        results_dict = compare(
            X=X,
            y=embedding,
            neighbors=3,
            comparison="KNN",
                    )
        results_dict["input_photo_coordinates"]=input[1]

        return JSONResponse(content=results_dict)


    except Exception as e:
        return Response(content=f"Error processing image: {str(e)}", media_type="text/plain", status_code=500)
