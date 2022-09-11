import streamlit as st
from clf import ImageClassifier
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Transformer model demo by Tidrael")
st.write("")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # image_byte = uploaded_file.getvalue()
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    with st.spinner("Loading model"):
        model = ImageClassifier()
    with st.spinner("Predict..."):
        prediction = model.predict(image=image)
    st.success(f"The result is: {prediction}")


