import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import re
import os

# =========================
# CONFIG STREAMLIT PAGE
# =========================
st.set_page_config(
    page_title="NutriAI - Analisis Gizi Makanan",
    page_icon="🍽️",
    layout="centered"
)

# =========================
# KONSTANTA GLOBAL
# =========================
TARGET_KAL_HARIAN = 2000  # asumsi kebutuhan energi harian rata-rata
MODEL_INPUT_SIZE = (320, 320)  # HARUS sama dengan input model saat training

# =========================
# STYLE (UI CSS)
# =========================
st.markdown(
    """
    <style>
    .macro-box {
        background-color: #1e1e2f;
        padding: 14px 16px;
        border-radius: 10px;
        color: #fff;
        font-size: 0.9rem;
        border: 1px solid #3a3a5c;
    }
    .macro-label {
        font-weight: 600;
        font-size: 0.8rem;
        opacity: 0.8;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: .05em;
    }
    .macro-value {
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .macro-desc {
        font-size: 0.8rem;
        opacity: 0.8;
        line-height: 1.2;
        margin-top: 2px;
    }
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 16px 20px;
        border: 1px solid #eee;
        box-shadow: 0 24px 48px rgba(0,0,0,.07);
    }
    .food-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        color: #222;
    }
    .food-sub {
        font-size: .9rem;
        color: #666;
        margin-top: 4px;
    }
    .confidence-chip {
        display: inline-block;
        background: #eef9f1;
        color: #0a7a2f;
        font-size: .75rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
        border: 1px solid #bfeccd;
        margin-top: 8px;
    }
    .section-header {
        font-size: .8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #666;
        margin-bottom: .5rem;
        margin-top: 1.5rem;
    }
    .cal-box {
        background: linear-gradient(135deg,#fff7ec,#fff);
        border: 1px solid #f4d3a0;
        border-radius: 12px;
        padding: 12px 16px;
    }
    .cal-val {
        font-size: 1.4rem;
        font-weight: 600;
        color: #b76b00;
        line-height: 1.2;
    }
    .cal-label {
        font-size: .8rem;
        font-weight: 500;
        color: #815100;
        opacity: .8;
    }
    .edu-box {
        background: #f5faff;
        border: 1px solid #cfe7ff;
        border-radius: 12px;
        padding: 12px 16px;
        color: #003562;
        font-size: .9rem;
        line-height: 1.4;
        box-shadow: 0 16px 32px rgba(0, 77, 255, .05);
    }
    .edu-title {
        font-weight: 600;
        font-size: .8rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #0062d1;
        margin-bottom: .25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# KELAS MAKANAN (HARUS SAMA URUTANNYA DENGAN TRAINING)
# =========================
class_names = [
    "Ayam Geprek (1 potong) = 394 kkal (62% lemak, 12% karb, 27% prot)",
    "Ayam Pop (1 potong) = 170 kkal (46% lemak, 11% karb, 43% prot)",
    "Ayam goreng (1 potong) = 391 kkal (50% lemak, 16% karb, 34% prot)",
    "Bakso (1 porsi) = 218 kkal (60% lemak, 15% karb, 25% prot)",
    "Batagor (1 porsi) = 400 kkal (43% lemak, 40% karb, 17% prot)",
    "Bika Ambon (1 potong) = 185 kkal (15% lemak, 80% karb, 5% prot)",
    "Cendol (1 porsi) = 465 kkal (36% lemak, 59% karb, 5% prot)",
    "Dadar Gulung (1 potong) = 139 kkal (43% lemak, 49% karb, 8% prot)",
    "Dendeng (1 porsi) = 123 kkal (57% lemak, 11% karb, 33% prot)",
    "Gorengan (1 potong) = 137 kkal (75% lemak, 19% karb, 6% prot)",
    "Gulai Ikan (1 porsi) = 241 kkal (42% lemak, 8% karb, 50% prot)",
    "Gulai Tambusu (1 porsi) = 204 kkal (39% lemak, 11% karb, 50% prot)",
    "Gulai Tunjang (1 porsi) = 243 kkal (42% lemak, 12% karb, 46% prot)",
    "Ikan Goreng (1 potong) = 192 kkal (23% lemak, 0% karb, 77% prot)",
    "Ketoprak (1 porsi) = 402 kkal (34% lemak, 50% karb, 16% prot)",
    "Klepon (1 buah) = 110 kkal (23% lemak, 72% karb, 5% prot)",
    "Kue Cubit (1 buah) = 70 kkal (14% lemak, 74% karb, 11% prot)",
    "Martabak Manis (1 potong) = 270 kkal (36% lemak, 54% karb, 10% prot)",
    "Martabak Telur (1 porsi) = 203 kkal (38% lemak, 41% karb, 22% prot)",
    "Mie Ayam (1 porsi) = 421 kkal (40% lemak, 44% karb, 16% prot)",
    "Nasi Goreng (1 porsi) = 250 kkal (34% lemak, 51% karb, 15% prot)",
    "Nasi Putih (1 porsi) = 135 kkal (2% lemak, 89% karb, 9% prot)",
    "Onde Onde (1 buah) = 101 kkal (16% lemak, 78% karb, 6% prot)",
    "Pempek (1 porsi) = 234 kkal (24% lemak, 49% karb, 26% prot)",
    "Pepes Ikan (1 potong) = 142 kkal (52% lemak, 0% karb, 48% prot)",
    "Pisang Ijo (1 porsi) = 188 kkal (37% lemak, 59% karb, 4% prot)",
    "Putu Ayu (1 buah) = 23 kkal (31% lemak, 57% karb, 12% prot)",
    "Rendang (1 porsi) = 468 kkal (51% lemak, 9% karb, 40% prot)",
    "Roti Bakar (1 potong) = 138 kkal (16% lemak, 68% karb, 16% prot)",
    "Sate Ayam (1 tusuk) = 34 kkal (58% lemak, 8% karb, 34% prot)",
    "Soto Ayam (1 porsi) = 312 kkal (44% lemak, 25% karb, 31% prot)",
    "Sup Ayam (1 porsi) = 75 kkal (29% lemak, 49% karb, 21% prot)",
    "Telur Balado (1 butir) = 71 kkal (73% lemak, 7% karb, 20% prot)",
    "Telur Dadar (1 porsi) = 93 kkal (71% lemak, 2% karb, 28% prot)",
    "Tempe Bacem (1 potong) = 49 kkal (54% lemak, 17% karb, 29% prot)"
]

# =========================
# FUNGSI LOAD MODEL AMAN
# =========================
@st.cache_resource(show_spinner="Memuat Model AI...")
def load_model_safe() -> tf.keras.Model:
    """Memuat model Keras dengan mencari beberapa candidate path."""
    candidate_paths = [
        "Training Dataset/model.keras",     # path yang ada di kode awal Anda
        "model.keras",                      # fallback kalau dipindahkan ke root
        "model_export/model.keras",         # fallback lain
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            # Penting: Pastikan versi TF di requirements.txt sama dengan versi saat training!
            return tf.keras.models.load_model(p)

    # Jika tidak ada file yang ditemukan, berikan pesan error
    st.error(
        "❌ **ERROR KRITIS: MODEL TIDAK DITEMUKAN.** Pastikan file 'model.keras' "
        "telah di-*upload* di lokasi yang benar (contoh: 'Training Dataset/model.keras')."
    )
    st.stop()

# Load model, aplikasi akan berhenti jika model gagal dimuat
model = load_model_safe()


# =========================
# PREPROCESS GAMBAR
# =========================
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Mengubah PIL Image menjadi NumPy array siap-input model (1, 320, 320, 3)."""
    img = pil_img.convert("RGB")
    img = img.resize(MODEL_INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# =========================
# PARSER LABEL -> DATA TERSTRUKTUR
# =========================
def parse_prediction_output(text: str) -> dict:
    """Menguraikan string label menjadi dictionary makronutrien."""
    try:
        food = re.search(r"^(.*?)\s*\(", text).group(1).strip()
        kalori = int(re.search(r"=\s*(\d+)\s*kkal", text).group(1))
        lemak = int(re.search(r"(\d+)%\s*lemak", text).group(1))
        karbo = int(re.search(r"(\d+)%\s*karb", text).group(1))
        protein = int(re.search(r"(\d+)%\s*prot", text).group(1))
    except Exception:
        # Fallback jika format regex gagal (misalnya, label error)
        food = text.split(" (")[0] if " (" in text else text
        kalori = 0
        lemak = karbo = protein = 0
        st.warning(f"Gagal mem-parse data nutrisi dari label: {text}")

    return {
        "food": food,
        "kalori": kalori,
        "lemak": lemak,
        "karbo": karbo,
        "protein": protein,
    }


# =========================
# KOMENTAR EDUKASI GIZI
# =========================
def nutrition_comment(kal: int, lemak: int, karbo: int, protein: int) -> str:
    """Memberikan komentar edukasi berdasarkan komposisi makro."""
    komentar = []

    if kal >= 400:
        komentar.append("Kalorinya tinggi per porsi, hati-hati kalau lagi defisit kalori.")
    elif kal <= 120:
        komentar.append("Kalorinya relatif rendah, cocok buat camilan ringan.")

    if lemak >= 50:
        komentar.append("Dominan lemak → kemungkinan digoreng, bersantan, atau berminyak.")
    if protein >= 25:
        komentar.append("Protein cukup tinggi, ini bantu rasa kenyang lebih lama.")
    if karbo >= 60:
        komentar.append("Karbohidrat dominan, jadi sumber energi cepat, tapi bisa naikin gula darah lebih cepat juga.")

    if not komentar:
        komentar.append("Komposisi makro cukup seimbang untuk satu kali makan normal.")

    return " ".join(komentar)


# =========================
# ANALISIS PORSI vs KEBUTUHAN HARIAN
# =========================
def portion_stats(kalori_per_unit: float) -> tuple[float, float]:
    """Menghitung persentase kebutuhan harian dan jumlah porsi yang dibutuhkan."""
    if kalori_per_unit <= 0:
        return 0.0, float("inf")

    pct_daily = (kalori_per_unit / TARGET_KAL_HARIAN) * 100.0
    how_many_for_daily = TARGET_KAL_HARIAN / kalori_per_unit
    return pct_daily, how_many_for_daily


# =========================
# INFERENSI MODEL
# =========================
def run_inference(pil_img: Image.Image) -> dict:
    """Menjalankan inferensi dan memproses hasilnya menjadi data terstruktur."""
    batch = preprocess_image(pil_img)
    probs = model.predict(batch)  # (1, num_classes)

    # ambil kelas dengan probabilitas tertinggi
    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])

    raw_label = class_names[pred_idx]
    parsed = parse_prediction_output(raw_label)

    # tambahkan info turunan
    parsed["confidence"] = confidence

    # edukasi nutrisi personal
    parsed["advice"] = nutrition_comment(
        parsed["kalori"], parsed["lemak"], parsed["karbo"], parsed["protein"]
    )

    # analisis kontribusi energi harian untuk 1 porsi dataset
    pct_daily, how_many_for_daily = portion_stats(parsed["kalori"])
    parsed["kalori_pct_daily"] = pct_daily
    parsed["how_many_for_daily"] = how_many_for_daily

    return parsed


# =========================
# HEADER / HERO
# =========================
st.markdown(
    """
    ### 🍽️ NutriAI
    Foto makanan → estimasi kalori & gizi per porsi dasar.
    _Ini hanya perkiraan. Untuk kondisi medis khusus, tetap konsultasi ahli gizi/dokter._
    """
)

# =========================
# INPUT GAMBAR
# =========================
uploaded_image = st.file_uploader(
    "📷 Upload foto makanan kamu (JPG / PNG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Gambar kamu", use_container_width=True)

    with st.spinner("Menganalisis gambar dan nutrisi..."):
        hasil = run_inference(img)

    # --- BLOK HASIL UTAMA ---
    st.markdown("#### 🔎 Hasil Deteksi")
    st.markdown(
        f"""
        <div class="result-card">
            <p class="food-title">{hasil['food']}</p>
            <p class="food-sub">
                Estimasi energi per porsi dataset:
                <b>{hasil['kalori']} kkal</b>
            </p>
            <div class="confidence-chip">
                Keyakinan model: {hasil['confidence']*100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- KONTRIBUSI ENERGI HARIAN ---
    st.markdown('<div class="section-header">Kontribusi energi harian</div>', unsafe_allow_html=True)

    daily_pct = hasil["kalori_pct_daily"]
    portion_need = hasil["how_many_for_daily"]

    colA, colB = st.columns([1, 3])
    with colA:
        st.markdown(
            f"""
            <div class="cal-box">
                <div class="cal-val">{daily_pct:.1f}%</div>
                <div class="cal-label">dari 2000 kkal/hari</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with colB:
        st.write(
            f"Satu porsi yang terdeteksi ≈ **{hasil['kalori']} kkal**, "
            f"yaitu sekitar **{daily_pct:.1f}%** dari asumsi kebutuhan energi harian {TARGET_KAL_HARIAN} kkal."
        )

        st.write(
            "Kalau (hipotetis) kamu cuma makan makanan ini saja seharian, "
            f"butuh sekitar **{portion_need:.1f}x porsi** ukuran dataset "
            f"untuk mencapai total ~{TARGET_KAL_HARIAN} kkal."
        )

        st.caption(
            "Definisi 1 porsi mengikuti dataset: contoh 1 tusuk sate ayam, "
            "1 potong ayam geprek, 1 buah klepon, 1 porsi bakso, dll. "
            "Kalau di foto ada banyak (misal 10 tusuk sate), total kalorinya = 10 × nilai di atas."
        )

    # --- KOMPOSISI MAKRO ---
    st.markdown('<div class="section-header">Komposisi makro</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="macro-box">
                <div class="macro-label">Lemak</div>
                <div class="macro-value">{hasil['lemak']}%</div>
                <div class="macro-desc">Sumber energi padat. Biasanya dari minyak goreng / santan.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(min(hasil['lemak'], 100) / 100)

    with col2:
        st.markdown(
            f"""
            <div class="macro-box">
                <div class="macro-label">Karbohidrat</div>
                <div class="macro-value">{hasil['karbo']}%</div>
                <div class="macro-desc">Bahan bakar cepat. Contoh: nasi, tepung, gula, mie.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(min(hasil['karbo'], 100) / 100)

    with col3:
        st.markdown(
            f"""
            <div class="macro-box">
                <div class="macro-label">Protein</div>
                <div class="macro-value">{hasil['protein']}%</div>
                <div class="macro-desc">Bangun & jaga otot. Contoh: ayam, telur, ikan.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(min(hasil['protein'], 100) / 100)

    # --- EDUKASI PERSONAL ---
    st.markdown('<div class="section-header">Catatan untuk kamu</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="edu-box">
            <div class="edu-title">Interpretasi Gizi</div>
            {hasil['advice']}
            <br><br>
            Tujuannya bukan untuk melarang makanan tertentu,
            tapi bantu kamu paham: ini seberapa padat kalori? lebih ke minyak? gula? protein?
            Jadi kamu bisa atur frekuensi & porsi, bukan sekadar 'boleh / gak boleh'.
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("⬆️ Upload dulu fotonya. Contoh: sate ayam, ayam geprek, rendang, klepon, ketoprak, dll.")
    st.caption("Tip: Foto jelas satu jenis makanan di tengah frame → hasil lebih akurat.")
