#image.open() and convert to RGB 
#all be functions 
#text is in a string format
#final response: [variable, 'modality']
#text=text
#image =img 
#pointcloud =pc 
#one function will return an output 
#use delay 
#send input back to output to test 
#point cloud can be visualized 
#read a random image 

import os
from alignment import *
from rag import *
import streamlit as st 
import pyvista as pv
import trimesh
import numpy as np
import time
from PIL import Image
import tempfile
from stpyvista import stpyvista  # Streamlit wrapper for PyVista
from streamlit_3d import streamlit_3d
import io
import uuid
import glob

#Path to 3d mesh file to test
obj_path = "/Data/ShapeNetSem/Files/models-OBJ/models/1004f30be305f33d28a1548e344f0e2e.obj"
checkpoint_path = "TrainedModels/ALIGN/final_template_30cat_fixed/100.pth" #"TrainedModels/ALIGN/subset_200_direct_new_loss_fixed_all/140.pth" #"TrainedModels/ALIGN/Baseline/150.pth"
dataset_path = "Data/ShapeNetSem/Datasets/final_template_30cat.csv"
img_dir = "Data/ShapeNetSem/Images/final_template_30cat/" #"Data/ShapeNetSem/Images/subset_200/"
pc_dir = "Data/ProcessedData/final_template_30cat_pc/" #"Data/ProcessedData/PointClouds/"
mesh_dir = "Data/ShapeNetSem/Files/models-OBJ/models/"
embed_path = f"Embeddings/ALIGN/final_template_30cat_fixed.pt" # f"Embeddings/ALIGN/subset_template_200.pt"
frontend_dir = "/Users/rishabhrao/Documents/VSCode/multimodal-3d-gen/env/lib/python3.12/site-packages/streamlit_3d/frontend/"
align_embd = 400

@st.cache_resource
def load_encoder(checkpoint_path, align_embd):
    return EncodeUserInputComplex(align_path=checkpoint_path, align_embd=align_embd)

@st.cache_resource
def load_cmr(dataset_path, embed_path):
    return CrossModalRetrival(dataset_path, embed_path)

encoder = load_encoder(checkpoint_path, align_embd)
cmr = load_cmr(dataset_path, embed_path)

def display_obj_as_3d(obj_file_path):
    # Full path to the GLB file you want to write to
    # Clean up old GLBs (optional)
    old_files = glob.glob(os.path.join(frontend_dir, "*.glb"))
    for file_path in old_files:
        try:
            os.remove(file_path)
        except:
            pass
    
    unique_filename = f"{uuid.uuid4().hex}.glb"
    glb_file_path = os.path.join(frontend_dir, unique_filename)

    # Export the GLB file (overwriting if needed)
    mesh = trimesh.load(obj_file_path)
    mesh.export(glb_file_path)

    # ✅ Ensure the file exists before calling the component
    if not os.path.exists(glb_file_path):
        st.warning("Waiting for GLB export...")
        st.stop()  # Halts execution until next rerun

    # ✅ Now load the model into the 3D viewer
    streamlit_3d(model=unique_filename, height=700, key=unique_filename)

#Function to load and visualize 3D mesh
def load_mesh(obj_file):
    try:
        mesh = trimesh.load(obj_file)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        vertices = np.array(mesh.vertices)
        faces = np.hstack([[len(face), *face] for face in mesh.faces]).astype(np.int64)
        pv_mesh = pv.PolyData(vertices, faces)
        pv_mesh.compute_normals()
        print('success')
        return pv_mesh
    except Exception as e:
        st.error(f"Error loading mesh: {e}")
        return None

#Function for backend processing
def process_input(input_data, input_modality, output_modality):
    """Simulate processing by sending input through a function."""
    print(type(input_data), input_modality, output_modality)
    # print(input_data, output_type)
    output = get_aligned_output_from_user_prompt(dataset_path, img_dir, mesh_dir, input_data, input_modality, output_modality, encoder, cmr)
    print(output)
    return output

# Streamlit UI
st.title("Multimodal 3D Generation")
st.subheader("Select Input Type")
col1, col2, col3 = st.columns(3)

if "input_type" not in st.session_state:
    st.session_state.input_type = None
if "output_type" not in st.session_state:
    st.session_state.output_type = None
if "button_highlight" not in st.session_state:
    st.session_state.button_highlight = False

if col1.button("Text", key="text_input_btn", use_container_width=True):
    st.session_state.input_type = "text"
    st.session_state.button_highlight = True
if col2.button("Image", key="image_input_btn", use_container_width=True):
    st.session_state.input_type = "img"
    st.session_state.button_highlight = True
if col3.button("3D Mesh", key="mesh_input_btn", use_container_width=True):
    st.session_state.input_type = "pc"
    st.session_state.button_highlight = True

input_type = st.session_state.input_type
input_data = None

if input_type:
    st.markdown(f"<div style='padding: 10px; background-color: #d3d3d3; border-radius: 5px; display: inline-block;'>Selected Input Type: <b>{input_type}</b></div>", unsafe_allow_html=True)

# Input Handling
if input_type == "text":
    input_data = st.text_area("Enter Text:")
elif input_type == "img":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg"], key="image_upload")
    if uploaded_file:
        input_data = Image.open(uploaded_file)
        st.image(input_data, caption="Uploaded Image", use_container_width=True)
elif input_type == "pc":
    uploaded_mesh = st.file_uploader("Upload a 3D Mesh (.npy)", type=["npy"], key="mesh_upload")
    if uploaded_mesh:
        input_data = np.load(uploaded_mesh)
        st.write("3D Mesh (Point Cloud) uploaded.")
        st.write(input_data.shape)

# Output Selection
st.subheader("Select Output Type")
col4, col5, col6 = st.columns(3)

if col4.button("Text Output", key="text_output_btn", use_container_width=True):
    st.session_state.output_type = "text"
    st.session_state.button_highlight = True
if col5.button("Image Output", key="image_output_btn", use_container_width=True):
    st.session_state.output_type = "img"
    st.session_state.button_highlight = True
if col6.button("3D Mesh Output", key="mesh_output_btn", use_container_width=True):
    st.session_state.output_type = "pc"
    st.session_state.button_highlight = True

output_type = st.session_state.output_type

if output_type:
     st.markdown(f"<div style='padding: 10px; background-color: #d3d3d3; border-radius: 5px; display: inline-block;'>Selected Output Type: <b>{output_type}</b></div>", unsafe_allow_html=True)

# Highlight Process Button
button_style = "background-color: #ffa500; color: white; font-size: 18px; padding: 10px 20px; border-radius: 5px;" if st.session_state.button_highlight else ""
st.markdown(f"""
    <style>
    .process-btn {{ {button_style} }}
    </style>
""", unsafe_allow_html=True)


# Process Button
st.markdown("""
    <style>
    .process-btn {
        background-color: black !important;
        color: white !important;
        font-size: 18px !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("Process Input", key="process_btn", help="Click to process input."):
    st.write(f"Processing...")
    time.sleep(10)  # Added 10-second delay before showing output
    retrieved_output = process_input(input_data, input_type, output_type)
    
    st.subheader("Output")
    if output_type == "text":
        st.success(f"Retrieved {input_type} into {output_type}: {retrieved_output}")
    elif output_type == "img":
        retrieved_img = Image.open(retrieved_output)
        st.image(retrieved_img, caption="Retrieved Image", use_container_width=True)
    elif output_type == "pc":
        st.write("Retrieved 3D Mesh")
        display_obj_as_3d(retrieved_output)