import streamlit as st
from PIL import Image
from ultralytics import YOLO
st.title("Rip current detection system using AI")
model = YOLO('model_yolo.pt')
confidence = float(st.slider("select confidence",10,20,90))/100

uploaded_file = st.file_uploader('insert image')

if uploaded_file is not None:
    source = Image.open(uploaded_file)
    st.image(source)
    button = st.button("predict")
    if button:
        res = model.predict(source,conf = confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:,:,::-1]

        st.image(res_plotted)




        


    

