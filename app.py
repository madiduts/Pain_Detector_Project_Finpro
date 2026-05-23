import streamlit as st
import pandas as pd
import joblib
from preprocess import extract_features

# Config halaman
st.set_page_config(
    page_title="PainSense AI | Deteksi Nyeri",
    page_icon="🩺",
    layout="wide", # Menggunakan layout lebar agar lebih profesional
    initial_sidebar_state="expanded"
)
# Load model dengan caching agar tidak perlu load ulang setiap kali ada interaksi
@st.cache_resource
def load_model():
    # Pastikan nama file sesuai dengan yang sudah dikompresi
    return joblib.load('nyeri_rf_baseline_model.pkl')

model = load_model()

# UI Sidebar
with st.sidebar:
    st.image("f5d358fa49587ddfc55b94d46af289b1.png", width=100)
    st.title("PainSense AI")
    st.markdown("---")
    st.markdown("**Informasi Sistem:**")
    st.markdown("Sistem *Machine Learning* ini menganalisis sinyal biosensor pasien untuk memprediksi tingkat nyeri secara objektif.")
    st.markdown("**Akurasi Model:** `80.0%`")
    st.markdown("**Algoritma:** `Random Forest`")
    st.markdown("---")
    st.markdown("👨‍⚕️ *Dikembangkan untuk Final Project 2026*")
    st.markdown("---")
    st.markdown("**Anggota Tim:**")
    st.markdown("1. Mahdi Imantaka Sutejo")
    st.markdown("2. Zulfi Alisya")
    st.markdown("3. Muhammad Fadhillah Nasri")

# Dashboard Utama
st.title("🩺 Dashboard Deteksi Nyeri Dini")
st.markdown("Unggah data CSV rekaman sensor fisiologis pasien (minimal 5 detik observasi / 20 baris data) untuk mendapatkan analisis tingkat keparahan nyeri.")

# Membagi layar menjadi 2 kolom (Kiri untuk Upload, Kanan untuk Hasil)
col_kiri, col_kanan = st.columns([1, 1.2])

with col_kiri:
    st.subheader("📂 Input Data Sensor")
    st.info("Format wajib: CSV dengan kolom hr, eda_tonic, eda_phasic, acc_x, acc_y, acc_z, bvp, temp.")
    uploaded_file = st.file_uploader("Seret dan lepas file CSV di sini", type=["csv"])

# Logika Prediksi dan Tampilan Hasil
with col_kanan:
    st.subheader("📊 Hasil Diagnosis AI")
    
    if uploaded_file is not None:
        try:
            # Membaca file
            df_pasien = pd.read_csv(uploaded_file)
            
            # Menampilkan indikator loading saat memproses
            with st.spinner("Memproses sinyal dan mengekstraksi fitur spasial-temporal..."):
                # Ekstraksi fitur
                fitur_model = extract_features(df_pasien)
                
                # Prediksi
                prediksi = model.predict(fitur_model)[0]
                
            # Menampilkan Hasil Berdasarkan Kelas (Warna Standar Triage Medis)
            if prediksi == 0:
                st.success("### 🟢 STATUS: LOW PAIN (Nyeri Ringan / Rileks)")
                st.markdown("**Rekomendasi:** Pasien dalam kondisi stabil. Tidak diperlukan intervensi analgesik tingkat tinggi saat ini.")
            
            elif prediksi == 1:
                st.warning("### 🟡 STATUS: MEDIUM PAIN (Nyeri Sedang)")
                st.markdown("**Rekomendasi:** Pasien menunjukkan indikasi rasa tidak nyaman secara fisiologis. Perlu observasi lanjutan dan kemungkinan pemberian pereda nyeri ringan.")
                
            elif prediksi == 2:
                st.error("### 🔴 STATUS: HIGH PAIN (Nyeri Hebat)")
                st.markdown("**Rekomendasi:** PERINGATAN KRITIS! Pasien mengalami tekanan nyeri akut. Segera berikan penanganan medis dan evaluasi dosis analgesik.")

            # Menampilkan Metrik Vital Pasien (Agar terlihat canggih)
            st.markdown("---")
            st.markdown("**Vital Sign Terakhir (Berdasarkan Data):**")
            
            metrik1, metrik2, metrik3 = st.columns(3)
            with metrik1:
                st.metric(label="Heart Rate", value=f"{df_pasien['hr'].iloc[-1]:.0f} BPM")
            with metrik2:
                st.metric(label="Temperature", value=f"{df_pasien['temp'].iloc[-1]:.1f} °C")
            with metrik3:
                st.metric(label="BVP Signal", value=f"{df_pasien['bvp'].iloc[-1]:.2f}")

            # Fitur Expander agar tabel mentah tidak menuh-menuhin layar
            with st.expander("Lihat Data Rekaman Mentah"):
                st.dataframe(df_pasien.tail(10))

        except Exception as e:
            st.error("Terjadi kesalahan pembacaan data. Pastikan format CSV sesuai.")
            st.exception(e)
    else:
        # Tampilan kosong sebelum data diunggah
        st.write("Menunggu data pasien diunggah...")
        st.image("https://cdn-icons-png.flaticon.com/512/2854/2854005.png", width=150)