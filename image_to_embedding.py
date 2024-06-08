from ultralytics import YOLO
import cv2
import os

import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType




model = YOLO('yolov8n.pt')
cars_img_list = ["img01","img02","img03","img04","img05","img06","img07","img08","img09","img10","img11","img12","img13","img14","img15"]
cars_cost_list = ["6.49","3.99","6.66","6.65","7.04","5.65","61.85","11.00","11.63","11.56","11.86","46.05","75.90","13.59","13.99"]


for im in cars_img_list:
    img = cv2.imread("cars_imgs/"+im+".jpg")
    img = cv2.resize(img,(320,245))

    results = model(img,stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),1)



            cropped_img = img[y1:y2, x1:x2]


        cv2.imwrite("cropped_imgs/"+im+"_cropped.jpg",cropped_img)


device = "cpu"

model_embed = imagebind_model.imagebind_huge(pretrained=True)
model_embed.eval()
model_embed.to(device)

embedding_list = []

for i in range(1,16):
    img_path = "cropped_imgs/img"+str(i)+"_cropped.jpg"
    print(img_path)
    vision_data = data.load_and_transform_vision_data([img_path], device)

    with torch.no_grad():
        image_embeddings = model_embed({ModalityType.VISION: vision_data})
    embedding_list.append(image_embeddings)


for i in embedding_list:
    print(i['vision'][0])
print("\n\n\n\n")


print(embedding_list[1]['vision'][0])




import pickle

with open('embedded_data.pickle', 'wb') as file:
    # Step 4: Use pickle.dump to serialize the data and write it to the file
    pickle.dump(embedding_list, file)
