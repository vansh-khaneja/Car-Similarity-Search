import streamlit as st
from PIL import Image
import base64
import os
from io import BytesIO  # Ensure BytesIO is imported

import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import pickle

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import cv2

import numpy as np
from qdrant_client.http.models import PointStruct  # Import PointStruct

from ultralytics import YOLO

# Set the title of the app
st.set_page_config(layout="wide")
st.title('Similar Cars Finder')
st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)
# Create a file uploader widget
uploaded_file = st.file_uploader("Upload an image of a car", type=["jpg", "jpeg", "png"])
device = "cpu"
model_embed = imagebind_model.imagebind_huge(pretrained=True)
model_embed.eval()
model_embed.to(device)


with open('embedded_data.pickle', 'rb') as file:
    # Use pickle.load to deserialize the data
    embedding_list = pickle.load(file)

print(embedding_list)

client = QdrantClient(":memory:")


client.recreate_collection(
    collection_name='vector_comparison',
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

client.upsert(
    collection_name='vector_comparison',
    points=[
        PointStruct(id=0, vector=embedding_list[0]['vision'][0].tolist()), # Use PointStruct to create objects
        PointStruct(id=1, vector=embedding_list[1]['vision'][0].tolist()),
        PointStruct(id=2, vector=embedding_list[2]['vision'][0].tolist()),
        PointStruct(id=3, vector=embedding_list[3]['vision'][0].tolist()),
        PointStruct(id=4, vector=embedding_list[4]['vision'][0].tolist()),
        PointStruct(id=5, vector=embedding_list[5]['vision'][0].tolist()),
        PointStruct(id=6, vector=embedding_list[6]['vision'][0].tolist()),
        PointStruct(id=7, vector=embedding_list[7]['vision'][0].tolist()),
        PointStruct(id=8, vector=embedding_list[8]['vision'][0].tolist()),
        PointStruct(id=9, vector=embedding_list[9]['vision'][0].tolist()),
        PointStruct(id=10, vector=embedding_list[10]['vision'][0].tolist()),
        PointStruct(id=11, vector=embedding_list[11]['vision'][0].tolist()),
        PointStruct(id=12, vector=embedding_list[12]['vision'][0].tolist()),
        PointStruct(id=13, vector=embedding_list[13]['vision'][0].tolist()),
        PointStruct(id=14, vector=embedding_list[14]['vision'][0].tolist()),

    ]
)



    
# Function to display images in a row with padding and price
def display_images_with_padding_and_price(images, prices, width, padding, gap):
    cols = st.columns(len(images))
    for col, img, price in zip(cols, images, prices):
        with col:
            col.markdown(
                f"""
                <div style="margin-right: {0}px; text-align: center;">
                    <img src="data:image/jpeg;base64,{img}" width="{250}px;margin-right: {50}px; ">
                    <p style="font-size: 20px;">₹{price} Lakhs</p>
                </div> 
                """,
                unsafe_allow_html=True,
            )









def image_to_similar_index(cv2Image):
        img = cv2.resize(cv2Image,(320,245))
        model = YOLO('yolov8n.pt')
        results = model(img,stream=True)
        results = model(img,stream=True)
        for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

                    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),1)



                    cropped_img = img[y1:y2, x1:x2]


                cv2.imwrite("test_cropped.jpg",cropped_img)
                
        vision_data = data.load_and_transform_vision_data(["test_cropped.jpg"], device)
        with torch.no_grad():
            test_embeddings = model_embed({ModalityType.VISION: vision_data})


        client.upsert(
                collection_name='vector_comparison',
                points=[
                  PointStruct(id=20, vector=test_embeddings['vision'][0].tolist()),
                ]
            )
        search_result = client.search(
            collection_name='vector_comparison',
            query_vector=test_embeddings['vision'][0].tolist(),
            limit=20 # Retrieve top similar vectors (excluding the new vector itself)
        )

        return [search_result[1].id,search_result[2].id,search_result[3].id,search_result[4].id]


cars_cost_list = ["5.12","6.89","6.66","7.28","8.54","14.35","17.42","8.45","13.63","22.47","15.86","16.42","38.43","13.59","13.99"]


if uploaded_file is not None:
    # Open and display the uploaded image
    car_image = Image.open(uploaded_file)
    img_array = np.array(car_image)
    st.image(car_image, caption='Uploaded Car Image', use_column_width=False, width=300)
    results = image_to_similar_index(img_array)
    print(results)

    # Directory where additional car images are stored
    car_images_dir = "C:/Users/VANSH KHANEJA/PROJECTS/test/code/cars_imgs"

    # Ensure the directory exists
    if os.path.exists(car_images_dir):
        car_images = [os.path.join(car_images_dir, img) for img in os.listdir(car_images_dir) if img.endswith(('jpg', 'jpeg', 'png'))]
        print(car_images)
    else:
        st.error(f"Directory {car_images_dir} does not exist")
        car_images = []

    # Check if there are enough images
    if len(car_images) < 4:
        st.error("Not enough car images in the local storage")
    else:
        car_imagess = []
        for i in results:
             car_imagess.append(car_images[i])
        car_prices = [cars_cost_list[a] for a in results]

        car_images_pil = []
        for img_path in car_imagess:
            try:
                img = Image.open(img_path)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                car_images_pil.append(img_str)
            except Exception as e:
                st.error(f"Error processing image {img_path}: {e}")

        if car_images_pil:
            st.subheader('Similar Cars with Prices')
            display_images_with_padding_and_price(car_images_pil, car_prices, width=200, padding=10, gap=20)