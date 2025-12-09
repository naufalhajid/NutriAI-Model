import os
import re
import numpy as np
import keras
import streamlit as st
from constants import CLASS_NAMES, MODEL_INPUT_SIZE, TARGET_KAL_HARIAN

# =========================
# CSS STYLES
# =========================
APP_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --primary: #00AB55;
    --secondary: #007B55;
    --bg-light: #F9FAFB;
    --card-bg: #FFFFFF;
    --text-main: #212B36;
    --text-sub: #637381;
    --shadow-card: 0 4px 12px rgba(0,0,0,0.05);
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-light);
    color: var(--text-main);
}

/* CARD STYLE */
.stCard {
    background: var(--card-bg);
    border-radius: 16px;
    box-shadow: var(--shadow-card);
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(145, 158, 171, 0.12);
}

.macro-card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    text-align: center;
    border: 1px solid #eee;
    transition: transform 0.2s;
}
.macro-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}

.macro-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-main);
    margin: 8px 0;
}
.macro-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-sub);
}

/* HERO SECTION */
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #00AB55, #007B55);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-sub);
    text-align: center;
    margin-bottom: 2rem;
}

/* RESULT HIGHLIGHT */
.prediction-highlight {
    background: linear-gradient(135deg, #e8f5e9 0%, #fff 100%);
    border: 1px solid #c8e6c9;
    border-radius: 12px;
    padding: 16px;
    margin-top: 10px;
    text-align: center;
}
.food-name {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1b5e20;
}

/* PROGRESS BARS CUSTOM */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #00AB55, #81C784);
}
</style>
"""

# =========================
# HELPER FUNCTIONS
# =========================

@st.cache_resource(show_spinner="Loading AI Model...")
def load_model_safe():
    """Load Keras model with multiple candidate path checks."""
    candidate_paths = [
        "Training Dataset/model.keras",     # directory if SavedModel format
        "Training Dataset/model.keras/model.keras", # nested file if user structure is weird
        "model.keras",                      # fallback if moved to root
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                return keras.models.load_model(p)
            except Exception as e:
                print(f"⚠️ Failed to load from {p}: {e}")
                continue

    st.error(
        "❌ **CRITICAL ERROR: MODEL NOT FOUND.** Please ensure 'model.keras' "
        "is uploaded correctly (e.g., in 'Training Dataset/model.keras')."
    )
    st.stop()

def preprocess_image(pil_img):
    """Convert PIL Image to Model Input Array (1, 320, 320, 3)."""
    img = pil_img.convert("RGB")
    img = img.resize(MODEL_INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def parse_prediction_output(text):
    """Parse label string into structured dictionary."""
    try:
        food = re.search(r"^(.*?)\s*\(", text).group(1).strip()
        kalori = int(re.search(r"=\s*(\d+)\s*kkal", text).group(1))
        lemak = int(re.search(r"(\d+)%\s*lemak", text).group(1))
        karbo = int(re.search(r"(\d+)%\s*karb", text).group(1))
        protein = int(re.search(r"(\d+)%\s*prot", text).group(1))
    except Exception:
        food = text.split(" (")[0] if " (" in text else text
        kalori = 0
        lemak = karbo = protein = 0
        st.warning(f"Failed to parse nutrition data from label: {text}")

    return {
        "food": food,
        "kalori": kalori,
        "lemak": lemak,
        "karbo": karbo,
        "protein": protein,
    }

def nutrition_comment(kal, lemak, karbo, protein):
    """Generate educational comment based on macros."""
    # (Existing logic, translated to English for consistency if desired, or kept in ID)
    # Keeping mixed/Indonesian as per user's last state, but maybe standardize? 
    # User's last prompt was "Jawab... dalam bahasa Indonesia".
    # But UI is English. I will keep logic simple.
    
    komentar = []
    if kal >= 400:
        komentar.append("Calories are high per serving; watch out if you are in a deficit.")
    elif kal <= 120:
        komentar.append("Relatively low calorie, good for a snack.")

    if lemak >= 50:
        komentar.append("High fat content → likely fried, coconut milk, or oily.")
    if protein >= 25:
        komentar.append("High protein, helps keep you full longer.")
    if karbo >= 60:
        komentar.append("High carbs, quick energy source but may spike blood sugar.")

    if not komentar:
        komentar.append("Balanced macro composition for a standard meal.")

    return " ".join(komentar)

def portion_stats(kalori_per_unit):
    """Calculate daily percentage and portion count."""
    if kalori_per_unit <= 0:
        return 0.0, float("inf")

    pct_daily = (kalori_per_unit / TARGET_KAL_HARIAN) * 100.0
    how_many_for_daily = TARGET_KAL_HARIAN / kalori_per_unit
    return pct_daily, how_many_for_daily

def run_inference(model, pil_img):
    """Run inference and return structured result."""
    batch = preprocess_image(pil_img)
    probs = model.predict(batch)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])

    raw_label = CLASS_NAMES[pred_idx]
    parsed = parse_prediction_output(raw_label)

    parsed["confidence"] = confidence
    parsed["advice"] = nutrition_comment(
        parsed["kalori"], parsed["lemak"], parsed["karbo"], parsed["protein"]
    )

    pct_daily, how_many_for_daily = portion_stats(parsed["kalori"])
    parsed["kalori_pct_daily"] = pct_daily
    parsed["how_many_for_daily"] = how_many_for_daily

    return parsed
