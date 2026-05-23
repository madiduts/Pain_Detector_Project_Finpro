# PainSense AI: Pendeteksi Tingkat Nyeri Berbasis Sinyal Fisiologis

🔗 **Live Demo:** https://paindetectorprojectfinpro-guzzxbwvklcstrzlqjjhvp.streamlit.app/

Sistem *machine learning* untuk memprediksi tingkat nyeri pasien (Low, Medium, High) secara objektif menggunakan data biosensor (*wearable devices*). 
Proyek ini dibangun untuk membantu observasi medis pada pasien yang kesulitan atau tidak mampu mengomunikasikan rasa sakit secara verbal (misalnya pasien kritis atau di ICU).

# 📌 Latar Belakang Masalah
Penilaian nyeri di rumah sakit umumnya masih subjektif (menggunakan skala angka 1-10 yang ditanyakan kepada pasien). Masalah muncul ketika pasien tidak sadar. 
Proyek ini mencoba mengukur nyeri berdasarkan respons sistem saraf otonom tubuh yang terekam lewat sensor:
* **EDA (Electrodermal Activity):** Mengukur konduktivitas keringat.
* **HR (Heart Rate):** Detak jantung.
* **BVP (Blood Volume Pulse) & Suhu Tubuh.**
* **Accelerometer (ACC):** Mengukur pergerakan fisik/kegelisahan pasien.

## 🔬 Metodologi & Ekstraksi Fitur
Data mentah dari sensor beresolusi 4Hz diproses menggunakan pendekatan observasi jeda waktu (Temporal Windowing) selama 5 detik (20 baris data).
* **Pemrosesan Sinyal EDA:** Menggunakan library `NeuroKit2` untuk memecah sinyal keringat menjadi `eda_tonic` (level dasar stres) dan `eda_phasic` (respons kejut/spontan).
* **Feature Engineering:** Menghitung *rolling mean*, *standard deviation*, dan *skewness* untuk menangkap tren perubahan sinyal dalam jendela 5 detik.
* **Strategi Validasi (Anti-Leakage):** Menggunakan `GroupShuffleSplit` berdasarkan ID Pasien. Model diuji pada subjek pasien yang benar-benar baru dan belum pernah dilihat saat fase *training*.

## 📊 Temuan Utama & Performa Model
Setelah melalui serangkaian eksperimen (*hyperparameter tuning*, SMOTE, XGBoost, hingga modifikasi *window size*), kami menetapkan **Random Forest** sebagai model akhir karena tingkat stabilitasnya yang paling rasional untuk skenario medis.

* **Akurasi Total:** 80%
* **Deteksi Nyeri Hebat (High Pain):** Sangat tangguh dengan **F1-Score 0.92**. Model sangat sensitif mendeteksi pasien yang berada dalam kondisi kritis.
* **Catatan Ilmiah pada "Medium Pain":** Kelas nyeri sedang (Medium) mentok di F1-Score ~0.36 terlepas dari algoritma apa pun yang digunakan.
  Eksperimen kami membuktikan bahwa ini bukan masalah ketidakseimbangan data (*class imbalance*), melainkan **Class Overlap (Aleatoric Uncertainty)** secara fisiologis.
  Tubuh manusia tidak memiliki batasan sinyal saraf yang tegas untuk mendefinisikan "nyeri sedang", polanya selalu bercampur antara rileks dan sangat nyeri.
