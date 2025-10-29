import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import zipfile

# Set page configuration
st.set_page_config(
    page_title="NutriAI: Klasifikasi Gambar Makanan",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="auto",
)

# Function to extract model from zip if needed
def extract_model_if_needed():
    """Extract model.tflite from zip file if it doesn't exist"""
    model_dir = 'Training Dataset'
    model_path = os.path.join(model_dir, 'model.tflite')
    zip_path = os.path.join(model_dir, 'model.zip')
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if model.tflite already exists
    if os.path.exists(model_path):
        return model_path
    
    # Check if zip file exists and extract it
    if os.path.exists(zip_path):
        try:
            with st.spinner('Mengekstrak model dari file zip...'):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                st.success('Model berhasil diekstrak!')
            return model_path
        except Exception as e:
            st.error(f"Error mengekstrak model: {e}")
            return None
    
    # If neither exists, return None
    return None if not os.path.exists(model_path) else model_path

# Function to load the TFLite model
@st.cache_resource
def load_model():
    try:
        model_path = extract_model_if_needed()
        
        if model_path is None or not os.path.exists(model_path):
            st.error("File model tidak ditemukan. Pastikan 'model.tflite' atau 'model.zip' ada di folder 'Training Dataset'.")
            return None
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to get class names and nutritional info
@st.cache_data
def get_class_names():
    dataset_path = 'Dataset Makanan New'
    
    if not os.path.exists(dataset_path):
        st.warning(f"Folder dataset '{dataset_path}' tidak ditemukan. Menggunakan nama kelas default.")
        return []
    
    try:
        class_names = sorted([
            d for d in os.listdir(dataset_path) 
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
        return class_names
    except Exception as e:
        st.error(f"Error membaca nama kelas: {e}")
        return []

# Function to preprocess the image
def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        img = image.resize((320, 320))
        img_array = np.array(img)
        
        # Ensure correct shape
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Add batch dimension and normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to parse nutritional info from class name
def parse_food_info(class_name):
    """Parse food name and nutritional information from class name"""
    food_name = class_name
    calories = "Informasi tidak tersedia"
    macros = "Informasi tidak tersedia"
    
    try:
        # Example format: "Bakso (1 Porsi = 357 Kalori (P: 25g, L: 20g, K: 18g))"
        if '(' in class_name:
            parts = class_name.split('(')
            food_name = parts[0].strip()
            
            # Extract nutritional info
            if len(parts) > 1:
                nutritional_info = '('.join(parts[1:]).rsplit(')', 1)[0]
                
                # Extract calories
                if '=' in nutritional_info and 'Kalori' in nutritional_info:
                    calories_part = nutritional_info.split('=')[1].split('(')[0].strip()
                    calories = calories_part
                
                # Extract macros (P, L, K)
                if 'P:' in nutritional_info:
                    macro_start = nutritional_info.find('(', nutritional_info.find('='))
                    if macro_start != -1:
                        macro_end = nutritional_info.find(')', macro_start)
                        if macro_end != -1:
                            macros = nutritional_info[macro_start+1:macro_end].strip()
    except Exception as e:
        st.warning(f"Gagal mem-parsing informasi nutrisi: {e}")
    
    return food_name, calories, macros

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

if uploaded_file is not None:
    if interpreter is None:
        st.error("Model tidak dapat dimuat. Silakan periksa file model dan coba lagi.")
    elif len(class_names) == 0:
        st.error("Nama kelas tidak dapat dimuat. Silakan periksa folder dataset.")
    else:
        # Display the uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)
            st.write("")
            st.write("Menganalisis gambar...")

            # Preprocess the image
            processed_image = preprocess_image(image)
            
            if processed_image is None:
                st.error("Gagal memproses gambar.")
            else:
                # Make prediction
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(input_details[0]['index'], processed_image)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])

                # Get the top prediction
                predicted_class_index = np.argmax(predictions)
                confidence = np.max(predictions)
                
                # Check if prediction index is valid
                if predicted_class_index >= len(class_names):
                    st.error(f"Index prediksi ({predicted_class_index}) melebihi jumlah kelas ({len(class_names)})")
                else:
                    predicted_class_name_full = class_names[predicted_class_index]
                    
                    # Parse the food name and nutritional info
                    food_name, calories, macros = parse_food_info(predicted_class_name_full)

                    # Display the prediction
                    st.subheader("Hasil Prediksi:")
                    st.success(f"**Makanan:** {food_name}")
                    st.info(f"**Kalori:** {calories}")
                    st.info(f"**Makro Nutrisi:** {macros}")
                    st.write(f"**Tingkat Keyakinan:** {confidence:.2%}")

                    # Show detailed predictions
                    with st.expander("Lihat Detail Prediksi"):
                        st.write("Model memberikan probabilitas berikut untuk setiap kelas:")
                        
                        # Create a dictionary of class names and their probabilities
                        prediction_dict = {}
                        for i in range(min(len(class_names), predictions.shape[1])):
                            food_name_only = class_names[i].split('(')[0].strip()
                            prediction_dict[food_name_only] = float(predictions[0][i])
                        
                        # Sort by probability and show top 10
                        sorted_predictions = dict(sorted(prediction_dict.items(), 
                                                        key=lambda x: x[1], 
                                                        reverse=True)[:10])
                        st.bar_chart(sorted_predictions)
        
        except Exception as e:
            st.error(f"Error saat memproses gambar: {e}")

elif interpreter is None and uploaded_file is None:
    st.info("👆 Silakan unggah gambar makanan untuk memulai analisis.")

# Sidebar
st.sidebar.header("Tentang NutriAI")
st.sidebar.info(
    "NutriAI adalah aplikasi berbasis AI untuk mengklasifikasikan gambar makanan Indonesia "
    "dan memberikan informasi nutrisi. Aplikasi ini dibuat menggunakan TensorFlow Lite dan Streamlit."
)

st.sidebar.header("Cara Penggunaan")
st.sidebar.markdown("""
1. Unggah gambar makanan (format: JPG, JPEG, atau PNG)
2. Tunggu model menganalisis gambar
3. Lihat hasil prediksi dan informasi nutrisi
4. Klik 'Lihat Detail Prediksi' untuk melihat probabilitas semua kelas
""")

st.sidebar.header("Requirements")
st.sidebar.code("""
streamlit
tensorflow
Pillow
numpy
""")
