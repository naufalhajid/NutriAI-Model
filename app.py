import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Try to import TensorFlow Lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    st.error("TensorFlow Lite runtime is required for this app.")
    raise

# Function to load the TFLite model
@st.cache_resource
def load_model():
    """Load the TFLite model from disk."""
    model_path = 'model.tflite'
    try:
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan. Pastikan '{model_path}' ada di direktori yang sama dengan app.py.")
            return None
        
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except tf.errors.OpError as e:
        st.error(f"Error saat memuat model TFLite. Pastikan model valid: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to parse nutritional info
def parse_nutrition_info(class_name):
    try:
        parts = class_name.split('(')
        food_name = parts[0].strip()

        # Extract nutritional info
        nutritional_info = None
        if len(parts) > 1:
            nutritional_info = '('.join(parts[1:]).rsplit(')', 1)[0]

        calories = None
        macros = None
        if nutritional_info:
            # Extract calories if available
            if '=' in nutritional_info:
                nutritional_split = nutritional_info.split('=')
                if len(nutritional_split) > 1 and 'Kalori' in nutritional_split[1]:
                    calories_part = nutritional_split[1].split('(')[0].strip()
                    calories = calories_part
                    
            # Extract macros if available
            if nutritional_info:
                macro_start = nutritional_info.find('Macronutrients')
                macro_end = nutritional_info.find(')', macro_start)
                if macro_start != -1 and macro_end != -1:
                    macros = nutritional_info[macro_start + len('Macronutrients'):macro_end].strip()
    except Exception as e:
        st.warning(f"Gagal mem-parsing informasi nutrisi untuk '{class_name}'. Menampilkan info default.")
    
    return food_name, calories, macros

# Example of how to use the model
if __name__ == "__main__":
    model = load_model()
    if model:
        # Prediction logic and image upload
        image = Image.open("example_image.jpg")  # Replace with actual image input
        # Assuming predictions and class_names are derived from your model
        predictions = np.array([0.1, 0.5, 0.4])  # Example prediction array
        class_names = ['Food A (Kalori=200, Macronutrients: Protein 20g, Carbs 30g)', 'Food B (Kalori=250, Macronutrients: Protein 25g, Carbs 40g)']
        predicted_class_index = np.argmax(predictions)
        food_name, calories, macros = parse_nutrition_info(class_names[predicted_class_index])

        st.write(f"Predicted food: {food_name}")
        st.write(f"Calories: {calories}")
        st.write(f"Macronutrients: {macros}")
