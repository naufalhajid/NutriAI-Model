# NutriAI: Model Klasifikasi Gambar Makanan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repositori ini berisi kode dan notebook untuk pengembangan, pelatihan, dan evaluasi model *deep learning* yang bertujuan untuk mengklasifikasikan gambar makanan. Model ini dibangun menggunakan arsitektur Convolutional Neural Network (CNN) dengan TensorFlow dan Keras. Tujuan utamanya adalah untuk menciptakan model yang akurat dan efisien yang dapat diimplementasikan pada perangkat mobile atau edge, sehingga model akhir diekspor ke format TensorFlow Lite (`.tflite`).

## Daftar Isi
- [Pendahuluan](#pendahuluan)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Model](#arsitektur-model)
- [Dataset](#dataset)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Cara Menjalankan](#cara-menjalankan)
- [Hasil dan Evaluasi](#hasil-dan-evaluasi)

## Pendahuluan
**NutriAI** adalah aplikasi cerdas yang menggabungkan *Computer Vision* dan *Generative AI* untuk membantu pengguna memahami kandungan gizi makanan mereka. 

Sistem ini tidak hanya mendeteksi jenis makanan dari gambar menggunakan **TensorFlow/Keras**, tetapi juga menyediakan konsultasi gizi interaktif melalui **Google Gemini AI**.

---

##  Fitur Utama

### 1.  Klasifikasi Makanan (Computer Vision)
- **Model Deep Learning**: Menggunakan arsitektur CNN (Convolutional Neural Network) yang dilatih khusus pada dataset makanan Indonesia.
- **Deteksi Otomatis**: Mengenali berbagai jenis makanan populer (misal: Sate Ayam, Rendang, Bakso, dll).
- **Estimasi Kalori & Makro**: Menampilkan estimasi energi (kkal), protein, lemak, dan karbohidrat per porsi standar.

### 2.  Konsultasi Gizi (AI Chatbot)
- **Integrasi Gemini AI**: Menggunakan model Google untuk analisis lebih dalam.
- **Tanya Jawab Interaktif**: Pengguna bisa bertanya, "Apakah makanan ini aman untuk diet keto?" atau "Bagaimana cara membakar kalori ini?".
- **Konteks Otomatis**: Chatbot otomatis "tahu" makanan apa yang baru saja Anda scan, sehingga jawaban lebih spesifik.

### 3.  Visualisasi & Edukasi
- **Progress Bar Makronutrisi**: Visualisasi komposisi lemak, karbo, dan protein.
- **Analisis Kebutuhan Harian**: Membandingkan kalori makanan dengan rata-rata kebutuhan harian (2000 kkal).

---
## Arsitektur Model
Model ini dibangun menggunakan Keras Sequential API dengan arsitektur CNN yang efektif untuk tugas klasifikasi gambar. Strukturnya terdiri dari beberapa lapisan konvolusi untuk ekstraksi fitur, lapisan pooling untuk reduksi dimensi, lapisan dropout untuk mencegah overfitting, dan diakhiri dengan lapisan dense untuk klasifikasi.

*(Catatan: Untuk detail arsitektur yang spesifik, silakan merujuk ke dalam file `NutriAI_Model.ipynb`.)*

## Dataset
Model ini dilatih menggunakan dataset gambar makanan.

- **Sumber Dataset**: Kaggle, dan dataset pribadi
- **Pra-pemrosesan**: Gambar di-rescale dan di-augmentasi (rotasi, zoom, flip horizontal) untuk meningkatkan ketahanan model.

---

## Teknologi yang Digunakan

- **Frontend**: [Streamlit](https://streamlit.io/) (Framework UI Python)
- **AI Core**:
  - **TensorFlow & Keras**: Untuk model klasifikasi gambar (`model.keras`).
  - **Google Generative AI (Gemini)**: Untuk fitur chatbot cerdas.
- **Data Processing**: NumPy, Pillow (PIL).

---

## Cara Menjalankan
1.  **Buka Jupyter Notebook**
    Jalankan perintah berikut di terminal Anda:
    ```sh
    jupyter notebook
    ```
2.  **Jalankan Notebook**
    Buka file `NutriAI_Model.ipynb` dan jalankan sel-sel kode secara berurutan dari atas ke bawah.
    - Pastikan path dataset sudah benar.
    - Anda dapat menyesuaikan *hyperparameter* seperti *learning rate*, jumlah *epoch*, atau *batch size* di dalam notebook.
3.  **Hasil Akhir**
    Setelah notebook selesai dijalankan, file `model.tflite` akan dihasilkan, yang merupakan model yang telah dikonversi dan siap untuk diimplementasikan.

## Hasil dan Evaluasi
Kinerja model dievaluasi pada dataset validasi untuk mengukur kemampuannya dalam melakukan generalisasi.

- **Akurasi & Loss**:
  <img width="1001" height="470" alt="Training and Validation" src="https://github.com/user-attachments/assets/a707387b-9cfa-4680-9555-ec3db3df1bdb" />

- **Classification Report**:
  Laporan ini memberikan rincian metrik *precision*, *recall*, dan *f1-score* untuk setiap kelas.
  <img width="880" height="762" alt="Classification Report" src="https://github.com/user-attachments/assets/c9156e70-a67d-454d-91fb-dbb6bb961515" />

- **Confusion Matrix**:
  Visualisasi ini membantu memahami kelas mana yang sering salah diklasifikasikan oleh model.
<img width="1407" height="1189" alt="confusin matrix" src="https://github.com/user-attachments/assets/ba714d99-2e18-4a54-a976-5b7a8b2aa4f9" />




