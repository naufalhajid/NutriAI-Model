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
    page_icon="🍽️",
    layout="centered"
)
st.markdown(APP_STYLE, unsafe_allow_html=True)

# =========================
# UI FUNCTIONS
# =========================

def render_sidebar():
    """Renders the sidebar settings."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Check for API Key in Streamlit Secrets
        env_api_key = st.secrets.get("GEMINI_API_KEY")
        
        if env_api_key:
            api_key = env_api_key
            st.success("✅ API Key loaded from environment!")
        else:
            api_key = st.text_input(
                "Gemini API Key", 
                type="password", 
                help="Enter your Google Gemini API Key to enable the AI Chatbot."
            )
                
        if api_key:
            genai.configure(api_key=api_key)
            if not env_api_key:
                st.success("✅ API Key connected!")
        else:
            st.warning("⚠️ Enter API Key to enable Chatbot.")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "**NutriAI** uses Deep Learning to recognize Indonesian food "
            "and Gemini AI for personalized nutrition consultation."
        )
    
    return api_key

def render_hero():
    """Renders the main hero section."""
    st.markdown('<h1 class="hero-title">NutriAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">AI-Powered Nutrition Analysis & Consultation</p>', unsafe_allow_html=True)

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
        st.markdown("### 📊 Nutrition Facts")
        
        daily_pct = hasil["kalori_pct_daily"]
        st.markdown(f"""
        <div style="background: #FFF3E0; padding: 20px; border-radius: 12px; border: 1px solid #FFE0B2; margin-bottom: 20px;">
            <div style="font-size: 0.9rem; color: #E65100; font-weight: 600; text-transform: uppercase;">Energy</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #E65100; line-height: 1;">{hasil['kalori']} <span style="font-size: 1rem;">kcal</span></div>
            <div style="font-size: 0.85rem; color: #EF6C00; margin-top: 8px;">
                ≈ {daily_pct:.1f}% of daily needs ({TARGET_KAL_HARIAN} kcal)
            </div>
            <div style="background: #FFCC80; height: 6px; border-radius: 3px; margin-top: 8px; width: 100%;">
                <div style="background: #EF6C00; height: 100%; border-radius: 3px; width: {min(daily_pct, 100)}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"💡 **AI Insight:** {hasil['advice']}")

    # Macros
    st.markdown("### 🥗 Macronutrients")
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
    st.markdown("---")
    st.subheader("💬 AI Nutritionist Chat")
    
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
        st.info(f"💡 You have **{remaining_quota}** questions remaining in this session.")
        chat_input_placeholder = "Ask about this food (e.g., 'Is this good for a keto diet?')"
        chat_disabled = False
    else:
        st.warning("⚠️ You have reached the maximum question limit for this session.")
        chat_input_placeholder = "Session limit reached."
        chat_disabled = True

    if prompt := st.chat_input(chat_input_placeholder, disabled=chat_disabled):
        if not api_key:
            st.error("❌ Please enter your Gemini API Key in the sidebar first.")
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
                            model_name='gemini-2.0-flash-lite',
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
                        st.error(f"❌ Gemini API Error: {e}")
    
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_count = 0
            st.rerun()

# =========================
# MAIN APP FLOW
# =========================
def main():
    api_key = render_sidebar()
    render_hero()
    
    # Load model once at start
    model = load_model_safe()
    
    uploaded_image = st.file_uploader(
        "📷 Upload food image (JPG / PNG)",
        type=["png", "jpg", "jpeg"]
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
                    <div class="warning-title">⚠️ Makanan Kurang Jelas / Tidak Dikenali</div>
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
        st.info("⬆️ Start by uploading a food photo above. Supported: JPG, PNG.")
        st.markdown("#### Try scanning:")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.caption("🍗 Fried Chicken"); st.caption("Satay")
        with c2: st.caption("🍜 Chicken Noodles"); st.caption("Meatballs")
        with c3: st.caption("🍲 Beef Rendang"); st.caption("Fried Rice")
        with c4: st.caption("🥞 Sweet Martabak"); st.caption("Omelette")

if __name__ == "__main__":
    main()


