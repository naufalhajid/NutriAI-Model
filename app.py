import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="NutriAI: Klasifikasi Gambar Makanan",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="auto",
)

# Function to load the TFLite model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('Training Dataset', 'model.tflite')
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to get class names and nutritional info
@st.cache_data
def get_class_names():
    class_names = sorted([d for d in os.listdir('Dataset Makanan New') if os.path.isdir(os.path.join('Dataset Makanan New', d))])
    return class_names

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((320, 320))
    img_array = np.array(img)
    if img_array.shape[2] == 4:  # Handle RGBA images
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

# Load model and class names
interpreter = load_model()
class_names = get_class_names()

# Streamlit App
st.title("🍔 NutriAI: Klasifikasi Gambar Makanan")
st.write(
    "Unggah gambar makanan untuk dianalisis. Model akan memprediksi jenis makanan "
    "dan menampilkan informasi nutrisi yang terkandung di dalamnya."
)

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("")
    st.write("Menganalisis gambar...")

    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the top prediction
    predicted_class_index = np.argmax(predictions)
    predicted_class_name_full = class_names[predicted_class_index]
    confidence = np.max(predictions)

    # Parse the food name and nutritional info
    food_name = predicted_class_name_full
    calories = "Informasi tidak tersedia"
    macros = ""
    try:
        # Example format: "Bakso (1 Porsi = 357 Kalori (P: 25g, L: 20g, K: 18g))"
        parts = predicted_class_name_full.split('(')
        if len(parts) > 1:
            food_name = parts[0].strip()
            # Nutritional info is in the last part
            nutritional_info = parts[-1].split(')')[0] # "1 Porsi = 357 Kalori (P: 25g, L: 20g, K: 18g)"
            calories = nutritional_info.split('=')[1].split('(')[0].strip() # "357 Kalori"
            macros = nutritional_info.split('(')[1].strip() # "P: 25g, L: 20g, K: 18g"
    except (IndexError, ValueError):
        # If parsing fails, use the full name and default values
        st.warning("Gagal mem-parsing informasi nutrisi dari nama kelas. Menampilkan nama lengkap.")

    # Display the prediction
    st.subheader("Hasil Prediksi:")
    st.success(f"**Makanan:** {food_name}")
    st.info(f"**Kalori:** {calories}")
    st.info(f"**Makro Nutrisi:** {macros}")
    st.write(f"**Tingkat Keyakinan:** {confidence:.2%}")

    with st.expander("Lihat Detail Prediksi"):
        st.write("Model memberikan probabilitas berikut untuk setiap kelas:")
        # Create a dictionary of class names and their probabilities
        prediction_dict = {class_names[i].split('(')[0].strip(): predictions[0][i] for i in range(len(class_names))}
        st.bar_chart(prediction_dict)


elif interpreter is None:
    st.error("Model tidak dapat dimuat. Silakan periksa file model dan coba lagi.")

st.sidebar.header("Tentang NutriAI")
st.sidebar.info(
    "NutriAI adalah aplikasi berbasis AI untuk mengklasifikasikan gambar makanan Indonesia "
    "dan memberikan informasi nutrisi. Aplikasi ini dibuat menggunakan TensorFlow Lite dan Streamlit."
)
