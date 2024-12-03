# Autores:
# - Alejandro Pastor Membrado
# - Ángel Romero Huici

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
from collections import Counter

import time

import streamlit as st


# SIFT
import cv2

# RESNET
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

resnet152 = models.resnet152(pretrained=True)
modules=list(resnet152.children())[:-1]
resnet152=nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False

#Efficientnet
efficientnet = models.efficientnet_b0(pretrained=True)
modules = list(efficientnet.children())[:-1]
efficientnet_model = nn.Sequential(*modules)
for p in efficientnet_model.parameters():
    p.requires_grad = False

# INCEPTION
from torchvision.models import inception_v3
inception = inception_v3(pretrained=True, transform_input=False)
inception.fc = nn.Identity()
inception.eval()

st.set_page_config(layout="wide")
device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())
# Path in which the images should be located
IMAGES_PATH = os.path.join(FILES_PATH, r'project\images')
# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, r'project')

classes = {
    'beaches' : list(range(0,90)),
    'bus' : list(range(90,180)),
    'dinosaurs' : list(range(180,270)),
    'elephants' : list(range(270,360)),
    'flowers' : list(range(360,450)),
    'foods' : list(range(450,540)),
    'horses' : list(range(540,630)),
    'monuments' : list(range(630,720)),
    'mountains_and_snow' : list(range(720,810)),
    'people_and_villages_in_Africa' : list(range(810,900)),
}

def get_class_name(number):
    """Devuelve el nombre de la clase asociada a un índice."""
    for class_name, numbers in classes.items():
        if number in numbers:
            return class_name
    return None
        
def get_result_classes(I):
    """Analiza las clases asociadas a los índices devueltos por una búsqueda. 
    Calcula un score relativo para cada clase, basado en su frecuencia respecto al total de clases identificadas."""
    match_classes = [get_class_name(i) for i in I]
    a = Counter(match_classes)
    scores = {}
    for k in a.keys():
        scores[k] = (a[k]/len(match_classes))*100
    return scores


def histo(img):
    """Convierte la imagen en una matriz plana y calcula un histograma para el rango de valores [0, 256].
    Se devuelve como un array de numpy"""
    return np.array([np.histogram(np.asarray(img).ravel(),256,[0,256])[0]]).astype('float32')

def sift(img):
    """Se inicializa un array de 0s con capacidad para almacenar hasta 128 descriptores. 
    La imagen se convierte a escala de grises  y los descriptores se obtienen 
    utilizando el operador SIFT previamente definido. 
    Finalmente, los descriptores generados se almacenan en el array y se devuelven."""
    xa = np.zeros((1,128)).astype('float32')
    sift = cv2.SIFT_create(nfeatures=128)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, mask=None)
    xa[0,:] = descriptors[0]
    return xa

def resnet(img):
    """Durante el preprocesado de la imagen, se redimensiona a 256px, se recorta el centro (224px), se convierte
    a tensor y se normaliza. Se pasa la imagen al modelo ResNet y se obtiene la salida de 2048 dimensiones.
    Este resultado se almacena en un array de numpy y se devuelve."""
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = img

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    features = resnet152(input_batch).numpy()[0,:,0,0]

    a = np.zeros((1,2048)).astype('float32')
    a[0,:] = features
    return a

def efficientnet(img):
    """Durante el preprocesado de la imagen, se redimensiona a 600px, se convierte
    a tensor y se normaliza. Se pasa la imagen al modelo EfficientNet, sin modificar los gradientes,
    y se obtiene la salida de 1028 dimensiones.
    A este resultado se le aplica otra normalización, se almacena en un array de numpy y se devuelve."""
    preprocess = transforms.Compose([
        transforms.Resize(600),  #Ajusta el tamaño al requerido
        transforms.CenterCrop(600),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = img
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torch.no_grad():
        features = efficientnet_model(input_tensor).cpu().numpy()[0, :, 0, 0]
    features = features / np.linalg.norm(features)

    a = np.zeros((1, 1280), dtype="float32")
    a[0, :] = features
    return a


def inception_v3(img):
    """Durante el preprocesado de la imagen, se redimensiona a 299px, se convierte
    a tensor y se normaliza. Se pasa la imagen al modelo InceptionV3, sin modificar los gradientes,
    y se obtiene la salida de 2048 dimensiones.
    A este resultado se le aplica otra normalización, se almacena en un array de numpy y se devuelve."""
    preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = img
    input_tensor = preprocess(input_image).unsqueeze(0)
    with torch.no_grad():
        features = inception(input_tensor).cpu().numpy()[0, :]
    features = features / np.linalg.norm(features)

    a = np.zeros((1, 2048), dtype="float32")
    a[0, :] = features
    return a

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    if (feature_extractor == 'Extractor 1: Histogramas de niveles de gris'):
        # Function to preprocess and extract features
        model_feature_extractor = histo
        indexer = faiss.read_index(os.path.join(DB_PATH,  'ext_1.index'))

    elif (feature_extractor == 'Extractor 2: SIFT'):
        model_feature_extractor = sift
        indexer = faiss.read_index(os.path.join(DB_PATH,  'ext_2.index'))

    elif (feature_extractor == 'Extractor 3: ResNet'):
        model_feature_extractor = resnet
        indexer = faiss.read_index(os.path.join(DB_PATH,  'ext_3.index'))
    
    elif (feature_extractor == 'Extractor 4: EfficientNet'):
        model_feature_extractor = efficientnet
        indexer = faiss.read_index(os.path.join(DB_PATH,  'ext_4.index'))

    elif (feature_extractor == 'Extractor 5: Inception V3'):
        model_feature_extractor = inception_v3
        indexer = faiss.read_index(os.path.join(DB_PATH,  'ext_5.index'))

    embeddings = model_feature_extractor(img_query)
    vector = np.float32(embeddings)
    faiss.normalize_L2(vector)

    _, indices = indexer.search(vector, k=n_imgs)

    return indices[0]

def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose feature extractor')
        option = st.selectbox('.', ('Extractor 1: Histogramas de niveles de gris', 'Extractor 2: SIFT', 'Extractor 3: ResNet', 'Extractor 4: EfficientNet', 'Extractor 5: Inception V3'))

        st.subheader('Upload image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file)
            st.image(img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('**Retrieving .......**')
            start = time.time()

            retriev = retrieve_image(img, option, n_imgs=11)
            end = time.time()
            st.markdown('**Finish in ' + str(end - start) + ' seconds**')

            st.markdown('**Classes found:**')
            result_classes = get_result_classes(retriev)
            for k in result_classes.keys():
                st.markdown(f'{k}: {str(result_classes[k])} %')

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, str(retriev[0]) + ".jpg"))
                st.image(image, use_container_width= 'always')

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, str(retriev[1])+ ".jpg"))
                st.image(image, use_container_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, str(retriev[u])+ ".jpg"))
                    st.image(image, use_container_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, str(retriev[u])+ ".jpg"))
                    st.image(image, use_container_width = 'always')

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, str(retriev[u])+ ".jpg"))
                    st.image(image, use_container_width = 'always')

if __name__ == '__main__':
    main()