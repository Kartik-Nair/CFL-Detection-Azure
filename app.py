import csv
import streamlit as st
import requests
import json
from PIL import Image
import os
import numpy as np
import tempfile
import shutil

from distance_calculator import calculate_distance

# from docker_utils import start_docker, cleanup
from wall_detector import detect_wall_edge


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


def adjust_pixels(pixels, resized_image_shape, input_image_dims):
    return (pixels / resized_image_shape[1]) * input_image_dims[1]


def calculate_distance_and_write_csv(
    core_seg_output_image,
    first_white_pixels_wall,
    last_white_pixels_wall,
    input_image_dims,
    mode,
    filename,
):
    top_distance, bottom_distance, top_core, bottom_core, top_wall, bottom_wall = (
        calculate_distance(
            np.array(core_seg_output_image),
            first_white_pixels_wall,
            last_white_pixels_wall,
        )
    )
    total_distance = top_distance + bottom_distance
    adjusted_top_distance = adjust_pixels(
        top_distance, resized_image_shape, input_image_dims
    )
    adjusted_bottom_distance = adjust_pixels(
        bottom_distance, resized_image_shape, input_image_dims
    )
    adjusted_total_distance = adjust_pixels(
        total_distance,resized_image_shape, input_image_dims
    )
    adjusted_top_wall = adjust_pixels(top_wall, resized_image_shape, input_image_dims)
    adjusted_bottom_wall = adjust_pixels(
        bottom_wall, resized_image_shape, input_image_dims
    )
    adjusted_top_core = adjust_pixels(top_core, resized_image_shape, input_image_dims)
    adjusted_bottom_core = adjust_pixels(
        bottom_core, resized_image_shape, input_image_dims
    )
    with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as tmpfile:
        writer = csv.writer(tmpfile)
        writer.writerow(
            [
                "Filename",
                "Total distance",
                "Top CFL distance",
                "Bottom CFL distance",
                "Top wall coordinates",
                "Bottom wall coordinates",
                "Top core coordinates",
                "Bottom core coordinates",
            ]
        )
        writer.writerow(
            [
                filename,
                str(adjusted_total_distance),
                str(adjusted_top_distance),
                str(adjusted_bottom_distance),
                str(adjusted_top_wall),
                str(adjusted_bottom_wall),
                str(adjusted_top_core),
                str(adjusted_bottom_core),
            ]
        )

    # If you want to save the temporary file to a specific path afterward
    csv_file_path = os.path.join(os.getcwd(), "output", "output.csv")
    shutil.move(tmpfile.name, csv_file_path)


# Function to display images based on dropdown selection
def display_images(selected_image):
    core_seg_output_image = detect_cfl(selected_image)
    wall_edge_output_image, _, _ = detect_wall_edge(selected_image)
    with col5:
        # Display input image
        # input_image_path = os.path.join(os.curdir, "input", selected_image)
        st.write("INPUT IMAGE")
        st.image(
            Image.open(selected_image).convert("RGB").resize((256, 256)),
            use_column_width=True,
        )
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

    if os.path.exists(os.path.join(os.getcwd() + "/output/output.csv")):
        os.remove(os.getcwd() + "/output/output.csv")
    for file in uploaded_files:
        input_image = Image.open(file).convert("RGB")
        print(input_image)
        input_image_dims = input_image.size
        input_image = input_image.resize((256, 256))
        core_seg_output_image = detect_cfl(file)
        wall_edge_output_image, first_white_pixels_wall, last_white_pixels_wall = (
            detect_wall_edge(file)
        )
        output_dir = os.path.join(os.getcwd() + "/output/")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        calculate_distance_and_write_csv(
            core_seg_output_image,
            first_white_pixels_wall,
            last_white_pixels_wall,
            input_image_dims,
            "a+",
            file.name,
        )
        core_output_filename = os.path.join(output_dir, f"core_{file.name}")
        wall_output_filename = os.path.join(output_dir, f"wall_{file.name}")

        core_seg_output_image.save(core_output_filename)
        wall_edge_output_image.save(wall_output_filename)


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
    shutil.make_archive(
        os.getcwd() + "/output", "zip", os.path.join(os.getcwd() + "/output/")
    )


def delete_archive_and_folder():
    os.remove(os.getcwd() + "/output.zip")
    shutil.rmtree(output_dir)
    st.stop()


if os.path.exists(os.getcwd() + "/output.zip"):
    with open(os.getcwd() + "/output.zip", "rb") as fp:
        btn = st.download_button(
            label="Download output",
            data=fp,
            file_name="output.zip",
            mime="application/zip",
            on_click=delete_archive_and_folder,
        )
