import streamlit as st
import torch
import os
import cv2
import base64  # Agregar esta línea de importación
import tempfile
import psutil
from obj_det_and_trk_streamlit_V2 import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------Main Function for Execution--------------------------
def main():

    st.image("detect.jpg", width=820)
    inference_msg = st.empty()
    st.sidebar.title("Configuración del usuario")

    input_source = "Video"  # Solo permitir cargar videos desde archivos locales

    conf_thres = st.sidebar.text_input("Umbral de confianza de las clases", "0.25")

    # Siempre guardar la salida del video y no mostrar las etiquetas
    save_output_video = 'Si'
    nosave = False
    display_labels = False

    weights = "yolov5n.pt"
    device = "cpu"

    # ------------------------- LOCAL VIDEO ------------------------
    if input_source == "Video":
        # Lista de nombres de clases en inglés del archivo YAML
        class_names = ['person', 'car', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign']
        selected_classes = st.sidebar.multiselect("Seleccionar clases a detectar", class_names)

        video = st.sidebar.file_uploader("Seleccionar video de entrada",
                                         type=["mp4", "avi"],
                                         accept_multiple_files=False)

        if st.sidebar.button("Iniciar seguimiento") and video is not None:
            # Convertir las clases seleccionadas a un tensor de PyTorch
            classes_tensor = torch.tensor([class_names.index(cls) for cls in selected_classes])
            stframe = st.empty()

            # Guardar el video cargado en el sistema de archivos temporal y obtener la ruta
            video_path = save_uploaded_file(video)

            # Llamar a la función detect y obtener el recuento de clases
            class_count = detect(weights=weights,
                                source=video_path,
                                stframe=stframe,
                                conf_thres=float(conf_thres),
                                device="cpu",
                                classes=classes_tensor,
                                names=class_names,
                                nosave=nosave,
                                display_labels=display_labels)

    torch.cuda.empty_cache()

# Guardar el archivo cargado en el sistema de archivos temporal y devolver la ruta del archivo
def save_uploaded_file(uploaded_file):
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_video.mp4"

# --------------------MAIN FUNCTION CODE------------------------
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass


# ------------------------------------------------------------------
