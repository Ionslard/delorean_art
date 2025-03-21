from photo_face_detection import delorean_photo_face_detection
from photo_face_embedding import embedding_image, delorean_normalisation
from comparison import compare
from show_comparison import show_comparison
import pandas as pd
import matplotlib.image as mpimg
from show_comparison import show_comparison


############################################################################################################
############################################################################################################
#                                       TEST

#on recupere la photo en np.ndarray dans le streamlit ?
photo=mpimg.imread("../etienne.jpg")
photo_face=delorean_photo_face_detection(photo) #on la crop
photo_emb_model = "VGG-Face"
y=delorean_normalisation(embedding_image(photo_face,photo_emb_model)) #on normalise et on l'embedd


emb_model="VGG-Face" #to edit
comparison="cosine" #to edit
neighbors=3 #to edit

#on charge le fichier d'embedding du modele choisi
X=pd.read_csv("processed_data/vgg_5000_6_7.csv",index_col=0) #to edit


# on compare et on renvoie le dictionnaire avec les images les plus proches
result=compare(X,y,neighbors,comparison,photo_face)
show_comparison(result) #on affiche les images les plus proches
