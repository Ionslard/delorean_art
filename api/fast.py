# TODO: Import your package, replace this by explicit imports of what you need
# from packagename.main import predict

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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
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
        # 2. embedder la partie croppée,
        # 3. comparer la matrice
        # 4. renvoyer l'image des tableaux "voisins" avec le titre et l'auteur


        # Encoder l'image pour la réponse > à supprimer quand le code sera complété
        _, img_encoded = cv2.imencode('.png', cv2_img)  # extension depends on which format is sent from Streamlit
        return Response(content=img_encoded.tobytes(), media_type="image/png")


    except Exception as e:
        return Response(content=f"Error processing image: {str(e)}", media_type="text/plain", status_code=500)


# # code initial
# # Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
# @app.get("/predict")
# def get_predict(input_one: float,
#             input_two: float):
#     # TODO: Do something with your input
#     # i.e. feed it to your model.predict, and return the output
#     # For a dummy version, just return the sum of the two inputs and the original inputs
#     prediction = float(input_one) + float(input_two)
#     return {
#         'prediction': prediction,
#         'inputs': {
#             'input_one': input_one,
#             'input_two': input_two
#         }
#     }
