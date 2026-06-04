import os
from PIL import Image
import streamlit as st
import google.generativeai as genai
# Import Custom Modules
from constants import TARGET_KAL_HARIAN, MAX_CHAT_QUESTIONS
from utils import APP_STYLE, load_model_safe, run_inference

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NutriAI - Analisis Gizi Makanan",
    page_icon="N",
    layout="centered"
)
st.markdown(APP_STYLE, unsafe_allow_html=True)

NUTRIAI_DESIGN = """
<style>
.main .block-container {
    max-width: 1080px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8FFFB 0%, #EFFAF5 100%);
    border-right: 1px solid rgba(16, 185, 129, 0.14);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #064E3B;
    letter-spacing: -0.02em;
}

.nutri-hero {
    background:
        radial-gradient(circle at 90% 12%, rgba(16, 185, 129, 0.18), transparent 28%),
        linear-gradient(135deg, #FFFFFF 0%, #ECFDF5 100%);
    border: 1px solid rgba(16, 185, 129, 0.18);
    border-radius: 28px;
    box-shadow: 0 30px 80px -45px rgba(6, 95, 70, 0.55);
    margin-bottom: 1.6rem;
    padding: clamp(1.5rem, 4vw, 2.4rem);
}

.nutri-eyebrow,
.section-eyebrow {
    color: #059669;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    margin-bottom: 0.75rem;
    text-transform: uppercase;
}

.nutri-title {
    color: #0F172A;
    font-size: clamp(2.4rem, 7vw, 4.2rem);
    font-weight: 900;
    letter-spacing: -0.06em;
    line-height: 0.96;
    margin: 0;
}

.nutri-subtitle {
    color: #475569;
    font-size: 1.05rem;
    line-height: 1.65;
    margin: 1.1rem 0 1.5rem;
    max-width: 720px;
}

.hero-pills,
.sample-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
}

.hero-pill,
.sample-chip {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid rgba(16, 185, 129, 0.18);
    border-radius: 999px;
    color: #065F46;
    display: inline-flex;
    font-size: 0.86rem;
    font-weight: 700;
    padding: 0.55rem 0.85rem;
}

.upload-panel,
.result-panel,
.chat-panel {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 24px;
    box-shadow: 0 20px 60px -42px rgba(15, 23, 42, 0.45);
    margin: 1rem 0 1.5rem;
    padding: clamp(1rem, 3vw, 1.4rem);
}

.section-title {
    color: #0F172A;
    font-size: 1.35rem;
    font-weight: 850;
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem;
}

.section-copy {
    color: #64748B;
    font-size: 0.96rem;
    line-height: 1.55;
    margin: 0 0 1rem;
}

.energy-card {
    background: linear-gradient(135deg, #FFF7ED 0%, #FFFFFF 100%);
    border: 1px solid #FED7AA;
    border-radius: 22px;
    margin-bottom: 1rem;
    padding: 1.2rem;
}

.energy-label {
    color: #C2410C;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.energy-value {
    color: #9A3412;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-top: 0.35rem;
}

.energy-note {
    color: #EA580C;
    font-size: 0.88rem;
    margin-top: 0.55rem;
}

.energy-track {
    background: #FED7AA;
    border-radius: 999px;
    height: 8px;
    margin-top: 0.8rem;
    overflow: hidden;
}

.energy-fill {
    background: linear-gradient(90deg, #F97316, #FB923C);
    border-radius: inherit;
    height: 100%;
}

.insight-card {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 18px;
    color: #166534;
    line-height: 1.55;
    padding: 1rem;
}

.stFileUploader {
    background: #FFFFFF;
    border: 1px dashed rgba(16, 185, 129, 0.35);
    border-radius: 20px;
    padding: 0.8rem;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 999px;
    font-weight: 800;
}

@media (max-width: 720px) {
    .hero-pills,
    .sample-grid {
        display: grid;
        grid-template-columns: 1fr;
    }
}
</style>
"""
st.markdown(NUTRIAI_DESIGN, unsafe_allow_html=True)

# =========================
# UI FUNCTIONS
# =========================

def render_sidebar():
    """Renders the sidebar settings."""
    with st.sidebar:
        st.title("NutriAI")
        st.caption("Food recognition and nutrition consultation for Indonesian meals.")
        
        # Check for API Key in Streamlit Secrets
        env_api_key = st.secrets.get("GEMINI_API_KEY")
        
        if env_api_key:
            api_key = env_api_key
            st.success("API key loaded from environment.")
        else:
            api_key = st.text_input(
                "Gemini API Key", 
                type="password", 
                help="Enter your Google Gemini API Key to enable the AI Chatbot."
            )
                
        if api_key:
            genai.configure(api_key=api_key)
            if not env_api_key:
                st.success("API key connected.")
        else:
            st.warning("Enter an API key to enable the chatbot.")
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "**NutriAI** uses Deep Learning to recognize Indonesian food "
            "and Gemini AI for personalized nutrition consultation."
        )
    
    return api_key

def render_hero():
    """Renders the main hero section."""
    st.markdown("""
    <section class="nutri-hero">
        <div class="nutri-eyebrow">Indonesian food intelligence</div>
        <h1 class="nutri-title">NutriAI</h1>
        <p class="nutri-subtitle">
            Upload a food photo, identify the dish, review nutrition estimates,
            and ask an AI nutritionist for practical guidance.
        </p>
        <div class="hero-pills">
            <span class="hero-pill">35 local food classes</span>
            <span class="hero-pill">Nutrition estimate</span>
            <span class="hero-pill">Gemini consultation</span>
        </div>
    </section>
    """, unsafe_allow_html=True)

def render_results(img, hasil):
    """Renders the analysis results."""
    # Layout: Image on Left, Key Stats on Right
    col_img, col_stats = st.columns([1, 1.2])
    
    with col_img:
        st.image(img, caption="Uploaded Image", use_container_width=True, clamp=True)
        st.markdown(f"""
        <div class="prediction-highlight">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">Detected:</div>
            <div class="food-name">{hasil['food']}</div>
            <div class="confidence-chip">Confidence: {hasil['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="section-eyebrow">Analysis result</div>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Nutrition facts</h2>', unsafe_allow_html=True)
        
        daily_pct = hasil["kalori_pct_daily"]
        st.markdown(f"""
        <div class="energy-card">
            <div class="energy-label">Energy</div>
            <div class="energy-value">{hasil['kalori']} <span style="font-size: 1rem;">kcal</span></div>
            <div class="energy-note">
                Approximately {daily_pct:.1f}% of daily needs ({TARGET_KAL_HARIAN} kcal)
            </div>
            <div class="energy-track">
                <div class="energy-fill" style="width: {min(daily_pct, 100)}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card">
            <strong>AI insight</strong><br>
            {hasil['advice']}
        </div>
        """, unsafe_allow_html=True)

    # Macros
    st.markdown('<div class="section-eyebrow">Macro composition</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Macronutrients</h2>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown(f"""
        <div class="macro-card">
            <div class="macro-label" style="color: #d32f2f;">Fat</div>
            <div class="macro-value">{hasil['lemak']}%</div>
            <div style="font-size: 0.8rem; color: #888;">Heavy energy</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(hasil['lemak'], 100) / 100)

    with m2:
        st.markdown(f"""
        <div class="macro-card">
            <div class="macro-label" style="color: #f57c00;">Carbs</div>
            <div class="macro-value">{hasil['karbo']}%</div>
            <div style="font-size: 0.8rem; color: #888;">Quick fuel</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(hasil['karbo'], 100) / 100)

    with m3:
        st.markdown(f"""
        <div class="macro-card">
            <div class="macro-label" style="color: #388e3c;">Protein</div>
            <div class="macro-value">{hasil['protein']}%</div>
            <div style="font-size: 0.8rem; color: #888;">Muscle builder</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(hasil['protein'], 100) / 100)

def render_chatbot(api_key, hasil):
    """Renders the chatbot section."""
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Ask follow-up questions</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">AI nutritionist chat</h2>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_count" not in st.session_state:
        st.session_state.chat_count = 0
    if st.session_state.get("chat_food") != hasil["food"]:
        st.session_state.messages = []
        st.session_state.chat_count = 0
        st.session_state.chat_food = hasil["food"]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    remaining_quota = MAX_CHAT_QUESTIONS - st.session_state.chat_count
    if remaining_quota > 0:
        st.info(f"You have **{remaining_quota}** questions remaining in this session.")
        chat_input_placeholder = "Ask about this food (e.g., 'Is this good for a keto diet?')"
        chat_disabled = False
    else:
        st.warning("You have reached the maximum question limit for this session.")
        chat_input_placeholder = "Session limit reached."
        chat_disabled = True

    if prompt := st.chat_input(chat_input_placeholder, disabled=chat_disabled):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Build system instructions containing the food data context
                        system_instruction = (
                            "Kamu adalah AI Ahli Gizi yang ramah dan berwawasan luas. "
                            "Tugasmu adalah membimbing pengguna memahami nutrisi makanan mereka.\n\n"
                            f"### Data Makanan yang Terdeteksi:\n"
                            f"Nama: **{hasil['food']}**\n"
                            f"- Kalori: {hasil['kalori']} kkal\n"
                            f"- Komposisi Makro: Lemak {hasil['lemak']}%, Karbo {hasil['karbo']}%, Protein {hasil['protein']}%\n"
                            f"- Kontribusi Harian: {hasil['kalori_pct_daily']:.1f}% dari kebutuhan energi harian rata-rata\n\n"
                            "Instruksi Jawaban:\n"
                            "1. Jawab pertanyaan pengguna secara langsung dan ringkas.\n"
                            "2. Berikan analisis singkat mengenai keseimbangan nutrisi makanan ini (apakah tinggi lemak/gula/protein?).\n"
                            "3. Berikan saran praktis (misal: 'cocok dimakan setelah olahraga' atau 'batasi porsinya').\n"
                            "4. Gunakan format markdown (bold, bullet points) agar mudah dibaca.\n"
                            "5. Hindari bahasa medis yang terlalu rumit."
                        )
                        
                        # Initialize model with system instruction
                        model_genai = genai.GenerativeModel(
                            model_name='gemini-1.5-flash-8b',
                            system_instruction=system_instruction
                        )
                        
                        # Reconstruct the conversation history in Gemini's format
                        formatted_history = []
                        for msg in st.session_state.messages[:-1]: # exclude current user message
                            formatted_history.append({
                                "role": "user" if msg["role"] == "user" else "model",
                                "parts": [msg["content"]]
                            })
                        
                        # Start chat with history and send message
                        chat = model_genai.start_chat(history=formatted_history)
                        response = chat.send_message(prompt)
                        
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        st.session_state.chat_count += 1
                    except Exception as e:
                        st.session_state.messages.pop()
                        st.error(f"Gemini API Error: {e}")
    
    if st.session_state.messages:
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.session_state.chat_count = 0
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MAIN APP FLOW
# =========================
def main():
    api_key = render_sidebar()
    render_hero()
    
    # Load model once at start
    model = load_model_safe()
    
    st.markdown("""
    <div class="upload-panel">
        <div class="section-eyebrow">Start analysis</div>
        <h2 class="section-title">Upload a food image</h2>
        <p class="section-copy">Use a clear, centered image of one dish for the best prediction quality.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Upload food image (JPG / PNG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        
        with st.spinner("Analyzing your food..."):
            hasil = run_inference(model, img)
        
        # Check confidence threshold (minimum 45% to display results)
        if hasil["confidence"] < 0.45:
            col_img, col_warn = st.columns([1, 1.2])
            with col_img:
                st.image(img, caption="Uploaded Image", use_container_width=True)
            with col_warn:
                st.markdown(f"""
                <div class="warning-card">
                    <div class="warning-title">Makanan Kurang Jelas / Tidak Dikenali</div>
                    <p style="color: #92400E; font-size: 0.95rem; margin-bottom: 12px;">
                        Tingkat keyakinan model hanya <b>{hasil['confidence']*100:.1f}%</b> (di bawah batas minimum 45%).
                    </p>
                    <p style="color: #B45309; font-size: 0.85rem; line-height: 1.4; text-align: left;">
                        Model mendeteksi kemiripan dengan <b>{hasil['food']}</b>, tetapi tingkat kepercayaan terlalu rendah untuk menyajikan informasi gizi yang akurat.
                    </p>
                    <p style="color: #B45309; font-size: 0.85rem; line-height: 1.4; text-align: left; margin-top: 8px;">
                        <b>Saran untuk hasil lebih baik:</b><br/>
                        • Pastikan pencahayaan terang dan tidak silau.<br/>
                        • Posisikan makanan di tengah kamera.<br/>
                        • Ambil gambar lebih dekat pada satu porsi makanan.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            render_results(img, hasil)
            render_chatbot(api_key, hasil)
        
    else:
        st.info("Start by uploading a food photo above. Supported formats: JPG and PNG.")
        st.markdown("""
        <div class="sample-grid">
            <span class="sample-chip">Fried chicken</span>
            <span class="sample-chip">Satay</span>
            <span class="sample-chip">Chicken noodles</span>
            <span class="sample-chip">Meatballs</span>
            <span class="sample-chip">Beef rendang</span>
            <span class="sample-chip">Fried rice</span>
            <span class="sample-chip">Sweet martabak</span>
            <span class="sample-chip">Omelette</span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


