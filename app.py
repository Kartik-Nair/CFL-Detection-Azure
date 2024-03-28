import csv
import streamlit as st
import requests
import sys
import json
from PIL import Image
import os
import numpy as np

from distance_calculator import calculate_distance
from docker_utils import start_docker, cleanup
from wall_detector import detect_wall_edge


# start_docker()
# Inject custom CSS with st.markdown to change the button color
st.markdown(
    """
<style>
           
button {
    background-color: #6A0DAD !important;
    color: white !important;
    border: 1px solid #6A0DAD;
}
button:hover {
    background-color: #5e0ca5 !important;
    border: 1px solid #5e0ca5;
}
</style>
""",
    unsafe_allow_html=True,
)

# Set the title of the app
st.header("CFL Detection App")

core_seg_output_image = None
wall_edge_output_image = None
uploaded_files = []

resized_image_shape = (256, 256)


# Function to perform CFL detection
def detect_cfl(selected_image):
    # Perform CFL detection on the provided image

    input_image = Image.open(selected_image).convert("RGB").resize((256, 256))

    image_array1 = np.array(input_image)
    image_array = np.expand_dims(image_array1, axis=0)

    # Creating JSON data for the request
    data = json.dumps(
        {"signature_name": "serving_default", "instances": image_array.tolist()}
    )

    # # Sending POST request to TensorFlow Serving API
    # url = "http://localhost:8501/v1/models/core_model:predict"
    # url = "https://cfl-detection-backend.azurewebsites.net/v1/models/core_model:predict"
    url = "http://20.116.219.85:8501/v1/models/core_model:predict"
    headers = {"content-type": "application/json"}
    response = requests.post(url, data=data, headers=headers)
    print(response)
    predictions = response.json()["predictions"]
    predictions_array = np.array(predictions)
    threshold = 0.9
    predictions_array[predictions_array >= threshold] = 1
    predictions_array[predictions_array < threshold] = 0
    predictions_array = predictions_array.reshape(256, 256)

    core_seg_output_image = Image.fromarray((predictions_array * 255).astype(np.uint8))
    return core_seg_output_image


def calculate_distance_and_write_csv(
    core_seg_output_image,
    first_white_pixels_wall,
    last_white_pixels_wall,
    input_image_dims,
    mode,
):
    distance = calculate_distance(
        np.array(core_seg_output_image), first_white_pixels_wall, last_white_pixels_wall
    )
    adjusted_distance = (distance / resized_image_shape[1]) * input_image_dims[1]
    csv_file_path = os.path.join(os.curdir + "/output/output.csv")
    with open(csv_file_path, mode=mode, newline="") as file:
        writer = csv.writer(file)
        writer.writerow(adjusted_distance)


# Function to display images based on dropdown selection
def display_images(selected_image):
    core_seg_output_image = detect_cfl(selected_image)
    wall_edge_output_image, _, _ = detect_wall_edge(selected_image)
    with col5:
        # Display input image
        # input_image_path = os.path.join(os.curdir, "input", selected_image)
        st.write("INPUT IMAGE")
        st.image(input_image, use_column_width=True)
    with col6:
        # Perform CFL detection
        st.write("CORE SEGMENTATION")
        st.image(core_seg_output_image, use_column_width=True)

    with col7:
        # Perform wall edge detection
        st.write("WALL EDGE DETECTION")
        st.image(wall_edge_output_image, use_column_width=True)


# Buttons for selecting input image
col1, col2 = st.columns(2)
with col1:
    input_type = st.radio("Select Input Type:", ("Single Image", "Batch Images"))

with col2:
    if input_type in "Single Image":
        file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if file:
            uploaded_files.append(file)
    elif input_type == "Batch Images":
        uploaded_files = st.file_uploader(
            "Choose input folder: ",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
        )


# Buttons for "Detect CFL" and "Save Output" side by side
col3, col4 = st.columns(2)
with col3:
    # if st.button("Detect CFL"):
    #     detect_cfl_clicked = True
    #     st.write("CFL Detection Initiated...")

    for file in uploaded_files:
        input_image = Image.open(file).convert("RGB")
        print(input_image)
        input_image_dims = input_image.size
        input_image = input_image.resize((256, 256))
        core_seg_output_image = detect_cfl(file)
        wall_edge_output_image, first_white_pixels_wall, last_white_pixels_wall = (
            detect_wall_edge(file)
        )
        output_dir = os.path.join(os.curdir + "/output/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, mode=0o077)
        calculate_distance_and_write_csv(
            core_seg_output_image,
            first_white_pixels_wall,
            last_white_pixels_wall,
            input_image_dims,
            "a+",
        )
        core_seg_output_image.save(output_dir + "core_" + file.name)
        wall_edge_output_image.save(output_dir + "wall_" + file.name)


col5, col6, col7 = st.columns(3)


# Get list of uploaded image names
if isinstance(uploaded_files, list):
    uploaded_images = [file for file in uploaded_files]
    uploaded_images_dict = {file.name: file for file in uploaded_files}
else:
    uploaded_images = []

# Dropdown to select image
selected_image_name = st.selectbox("Select Image", list(uploaded_images_dict.keys()))
if len(uploaded_images_dict) > 0:
    selected_image = uploaded_images_dict[selected_image_name]
else:
    selected_image = None

# Display images based on selected image
if selected_image:
    display_images(selected_image)


# if st.button("Clean container for exit"):
#     cleanup()
#     sys.exit()
