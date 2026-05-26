import os
import re
import numpy as np
import keras
import streamlit as st
from constants import CLASS_NAMES, MODEL_INPUT_SIZE, TARGET_KAL_HARIAN

MODEL_CANDIDATE_PATHS = (
    "Training Dataset/model.keras",
    "Training Dataset/model.keras/model.keras",
    "model.keras",
)

# =========================
# CSS STYLES
# =========================
APP_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #10B981;
    --primary-glow: rgba(16, 185, 129, 0.15);
    --secondary: #059669;
    --bg-gradient: linear-gradient(135deg, #F0FDFA 0%, #FFF 100%);
    --card-bg: rgba(255, 255, 255, 0.85);
    --text-main: #0F172A;
    --text-sub: #475569;
    --shadow-premium: 0 10px 30px -10px rgba(0, 168, 89, 0.08), 0 1px 3px rgba(0,0,0,0.02);
    --border-color: rgba(16, 185, 129, 0.12);
}

/* Base Body Override */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg-gradient) !important;
    color: var(--text-main) !important;
}

/* Premium Card Style with Glassmorphism */
.stCard, div[data-testid="stMetricValue"] {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-premium);
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

/* Macro Cards Styling */
.macro-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    padding: 22px 16px;
    border-radius: 18px;
    box-shadow: 0 4px 20px -5px rgba(0,0,0,0.04);
    text-align: center;
    border: 1px solid rgba(226, 232, 240, 0.8);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.macro-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px -10px rgba(16, 185, 129, 0.25);
    border-color: rgba(16, 185, 129, 0.3);
}

.macro-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-main);
    margin: 8px 0;
    background: linear-gradient(135deg, #0F172A 0%, #334155 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.macro-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-sub);
}

/* Hero Section Banner */
.hero-title {
    font-size: 3rem;
    font-weight: 850;
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
    animation: fadeIn 0.8s ease-out;
}
.hero-sub {
    font-size: 1.1rem;
    color: var(--text-sub);
    text-align: center;
    margin-bottom: 2.5rem;
    font-weight: 500;
    animation: fadeIn 1s ease-out;
}

/* Result Prediction Box */
.prediction-highlight {
    background: linear-gradient(135deg, rgba(209, 250, 229, 0.45) 0%, rgba(255, 255, 255, 0.9) 100%);
    border: 1.5px solid rgba(16, 185, 129, 0.25);
    border-radius: 18px;
    padding: 20px;
    margin-top: 15px;
    text-align: center;
    box-shadow: 0 10px 25px -15px rgba(16, 185, 129, 0.2);
    backdrop-filter: blur(8px);
}
.food-name {
    font-size: 2.1rem;
    font-weight: 800;
    color: #065F46;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}
.confidence-chip {
    display: inline-block;
    background: #D1FAE5;
    color: #065F46;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 6px 16px;
    border-radius: 9999px;
    box-shadow: 0 4px 10px -3px rgba(16, 185, 129, 0.15);
}

/* Alert/Warning Card for low confidence */
.warning-card {
    background: linear-gradient(135deg, rgba(254, 243, 199, 0.45) 0%, rgba(255, 255, 255, 0.9) 100%);
    border: 1.5px solid rgba(245, 158, 11, 0.3);
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 10px 25px -15px rgba(245, 158, 11, 0.15);
    backdrop-filter: blur(8px);
}
.warning-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #92400E;
    margin-bottom: 8px;
}

/* Progress Bar Custom Styling */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #10B981, #34D399) !important;
    border-radius: 9999px;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""

# =========================
# HELPER FUNCTIONS
# =========================

@st.cache_resource(show_spinner="Loading AI Model...")
def load_model_safe():
    """Load Keras model with multiple candidate path checks."""
    for p in MODEL_CANDIDATE_PATHS:
        if os.path.exists(p):
            try:
                model = keras.models.load_model(p)
                validate_model_output(model, p)
                return model
            except Exception as e:
                print(f"⚠️ Failed to load from {p}: {e}")
                continue

    st.error(
        "❌ **CRITICAL ERROR: MODEL NOT FOUND.** Please ensure 'model.keras' "
        "is uploaded correctly (e.g., in 'Training Dataset/model.keras')."
    )
    st.stop()

def validate_model_output(model, model_path):
    """Stop early when the loaded model does not match the app labels."""
    output_shape = getattr(model, "output_shape", None)
    if not output_shape:
        return

    output_classes = output_shape[-1]
    expected_classes = len(CLASS_NAMES)
    if output_classes is None or int(output_classes) == expected_classes:
        return

    st.error(
        "❌ **MODEL / LABEL MISMATCH.** "
        f"Model `{model_path}` outputs {output_classes} classes, "
        f"but `CLASS_NAMES` contains {expected_classes} labels."
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
    """Parse label string into structured dictionary, looking up constants.NUTRITION_DB."""
    from constants import NUTRITION_DB
    if text in NUTRITION_DB:
        return NUTRITION_DB[text].copy()

    # Fallback to regex parser in case of anomalies
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
    if pred_idx >= len(CLASS_NAMES):
        st.error(
            "❌ Model prediction index is outside `CLASS_NAMES`. "
            "Please verify the model artifact and label ordering."
        )
        st.stop()

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
