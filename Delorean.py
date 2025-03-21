import streamlit as st
from PIL import Image
import time



# Streamlit app interface
def main():

    st.markdown(
        """
    <style>
    .stApp {
        background-color:  #8294C4 ;
    }

      .stTitle {

        background-color:  #ACB1D6 ;
    }

    </style>
    """,
    unsafe_allow_html=True)


    col1, col2 = st.columns([1, 1])

    with col1:
        st.title("DeLorean Art: 🎨 ")
        st.markdown("### Voyage dans le temps à travers de l'art")

    with col2:
        image = Image.open("img/Delorean.png")
        st.image(image, use_container_width=True)



    consent = st.checkbox("I consent to the use of face detection for my uploaded image")
    uploaded_file = st.file_uploader("Please upload your image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and consent:
        with st.spinner('Processing your image, please wait...'):
            time.sleep(4)  # Simulate processing time (4 seconds)

        image_paths = [
            uploaded_file,
            "processed_data/game_dataset_delorean_art_model_7/richard-whitney_allen-kenney_face_1.jpg",
            "processed_data/game_dataset_delorean_art_model_7/richard-whitney_allen-kenney_detected.jpg",
            "processed_data/game_dataset_delorean_art_model_7/richard-whitney_nancy_face_1.jpg",
            "processed_data/game_dataset_delorean_art_model_7/richard-whitney_nancy_detected.jpg",
            "processed_data/game_dataset_delorean_art_model_7/albert-huie_the-counting-lesson-1938_face_1.jpg",
            "processed_data/game_dataset_delorean_art_model_7/albert-huie_the-counting-lesson-1938_detected.jpg",
            "processed_data/game_dataset_delorean_art_model_7/yayoi-kusama_watching-the-sea-1989_detected.jpg",
            "processed_data/game_dataset_delorean_art_model_7/keisai-eisen_pipe-smokers-1835(1)_face1.jpg",
            "processed_data/game_dataset_delorean_art_model_7/akseli-gallen-kallela_symposium-1894_face_1.jpg",
            "processed_data/game_dataset_delorean_art_model_7/yayoi-kusama_watching-the-sea-1989_detected.jpg",
            "processed_data/game_dataset_delorean_art_model_7/keisai-eisen_pipe-smokers-1835(1)_detected.jpg"
    ]
        st.write("### Face Detection Results")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("**Input Image**")
            st.image(image_paths[0], width=300,use_container_width=True)
            st.write("-----")

        with col2:
            st.write("**Face 1**")
            st.image(image_paths[1],  width=300,use_container_width=True )
            st.write("Original Artwork 1")
            st.image(image_paths[2],  width=300,use_container_width=True )


        with col3:
            st.write("**Face 2**")
            st.image(image_paths[3], width=300,use_container_width=True)
            st.write("Original Artwork 2")
            st.image(image_paths[4],  width=300,use_container_width=True )


        with col4:
            st.write("**Face 3**")
            st.image(image_paths[5], width=300, use_container_width=True)
            st.write("Original Artwork 3")
            st.image(image_paths[6], width=300, use_container_width=True)
    elif uploaded_file is not None and not consent:
        st.warning("You need to consent to face detection to see the results.")


if __name__ == "__main__":
    main()
