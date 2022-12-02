import streamlit as st

import torchvision
from torchvision import io
import numpy as np
import torch

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import inception_v3, Inception_V3_Weights

from PIL import Image, ImageOps
import json
import cv2


def create_model(name):
    if name == 'inception_v3':
        model = torch.load('inception_v3_model.pt')
    if name == 'resnet18':
        model = torch.load('resnet18_model.pt')
    return model


def image_preprocessing(image):
    image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
    tensor = torchvision.transforms.ToTensor()
    image = tensor(image)
    return image

def get_black_and_white_pic(img):
    converted_img = np.array(img.convert('RGB'))
    gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
    slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
    st.image(blackAndWhiteImage)



def main_page():
    st.markdown('''<h1 style="text-align: center; font-family: 'Gill Sans'; color: #D8D8D8"
        >Привет 👋 
        На связи классификатор!</h1>''', 
        unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.image('https://miro.medium.com/max/2930/1*Y40V8ZZ9T_XI-eGQulwIRQ.png')
    st.markdown('''<p style="text-align: center; font-family: 'Gill Sans'; font-size: 26px; color: #D8D8D8">Я помогу тебе 
                тебе определить, что изображено на картинке </p>''', 
            unsafe_allow_html=True)
    st.markdown('''<p style="text-align: center; font-family: 'Gill Sans'; font-size: 100px; color: #D8D8D8">
            😃 😁 😉</p>''', 
    unsafe_allow_html=True)
    st.markdown('''<p style="text-align: center; font-family: 'Gill Sans'; font-size: 20px; color: #D8D8D8">
                Выберите в сайдбаре слева 👈, что хотите классифицировать</p>''', 
        unsafe_allow_html=True)

labels = json.load(open('imagenet_class_index.json'))
decoder = lambda x: labels[str(x)][1]
        


def get_random_classification():
    
    st.markdown('''<h1 style="text-align: center; font-family: 'Gill Sans'; color: #D8D8D8"
        >Загрузите любое изображение, чтобы узнать, что на нем изображено</h1>''', 
        unsafe_allow_html=True)
    model = create_model('inception_v3')
    # Загрузка изображения
    file = st.file_uploader(f'Пожалуйста, загрузите изображение', type=["jpg", "png"])
    
    if file is not None:
        model.eval()
        image = Image.open(file)
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.image(image)
        with col2:
            get_black_and_white_pic(image)
        image = image_preprocessing(image)
        st.subheader(f'Это больше всего похоже на {decoder(model(image.unsqueeze(0)).argmax().item())}')


def get_cats_and_dogs_classification():
    
    st.markdown('''<h1 style="text-align: center; font-family: 'Gill Sans'; color: #D8D8D8"
    >Попробуйте классифицировать котиков и собачек</h1>''', 
    unsafe_allow_html=True)
    
    model = create_model('resnet18')
    
    file = st.file_uploader(f'Пожалуйста, загрузите изображение', type=["jpg", "png"])
    
    if file is not None:
        model.eval()
        image = Image.open(file)
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.image(image)
        with col2:
            get_black_and_white_pic(image)
        image = image_preprocessing(image)
        st.subheader(f'Это больше всего похоже на {decoder(model(image.unsqueeze(0)).argmax().item())}')


page_names_to_funcs = {
    "Приветствие": main_page,
    "Классификация любых изображений": get_random_classification,
    "Классификация котиков и собачек": get_cats_and_dogs_classification
}
    
        
selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


        
#if __name__ == '__main__':
#    main_page()