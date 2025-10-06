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
Dalam pengembangan sistem pengenalan gambar, klasifikasi gambar makanan merupakan salah satu aplikasi yang populer dan menantang. Proyek ini berfokus pada pembangunan dan evaluasi sebuah model *deep learning* yang dapat mengenali dan mengklasifikasikan gambar makanan ke dalam berbagai kategori (misalnya, "nasi goreng", "sate", "salad", dll.).

Tantangan utama setelah melatih model adalah memastikan kinerjanya tidak hanya baik pada data latih, tetapi juga dapat melakukan generalisasi dengan baik pada data baru yang belum pernah dilihat sebelumnya. Oleh karena itu, evaluasi mendalam menjadi fokus penting dalam proyek ini untuk memastikan keandalan model.

## Fitur Utama
- **Klasifikasi Multi-Kelas**: Model mampu mengklasifikasikan gambar ke dalam beberapa kategori makanan yang berbeda.
- **Arsitektur CNN**: Menggunakan arsitektur Convolutional Neural Network yang efektif untuk tugas pengenalan gambar.
- **Augmentasi Data**: Menerapkan teknik augmentasi gambar untuk meningkatkan variasi data latih dan mencegah *overfitting*.
- **Evaluasi Kinerja**: Analisis performa model menggunakan metrik standar seperti *accuracy*, *precision*, *recall*, dan *F1-score*, serta visualisasi *confusion matrix*.
- **Siap untuk Deployment**: Model akhir dikonversi ke format `.tflite` untuk optimasi dan kemudahan implementasi pada perangkat Android, iOS, atau perangkat IoT lainnya.

## Arsitektur Model
Model ini dibangun menggunakan Keras Sequential API dengan arsitektur CNN yang efektif untuk tugas klasifikasi gambar. Strukturnya terdiri dari beberapa lapisan konvolusi untuk ekstraksi fitur, lapisan pooling untuk reduksi dimensi, lapisan dropout untuk mencegah overfitting, dan diakhiri dengan lapisan dense untuk klasifikasi.

*(Catatan: Untuk detail arsitektur yang spesifik, silakan merujuk ke dalam file `NutriAI_Model.ipynb`.)*

## Dataset
Model ini dilatih menggunakan dataset gambar makanan.

- **Sumber Dataset**: Kaggle, dan dataset pribadi
- **Pra-pemrosesan**: Gambar di-rescale dan di-augmentasi (rotasi, zoom, flip horizontal) untuk meningkatkan ketahanan model.

## Teknologi yang Digunakan
- **Python 3.x**
- **TensorFlow & Keras**: Framework utama untuk membangun dan melatih model *deep learning*.
- **Scikit-learn**: Untuk metrik evaluasi seperti *classification report* dan *confusion matrix*.
- **NumPy**: Untuk operasi numerik.
- **Matplotlib & Seaborn**: Untuk visualisasi data, seperti plot histori pelatihan dan *confusion matrix*.
- **Jupyter Notebook / Google Colab**: Untuk lingkungan pengembangan interaktif.

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


