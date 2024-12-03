import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Judul aplikasi web
st.title("Aplikasi Prediksi Harga Laptop")

# Menu sidebar untuk memilih antara Overview, Data Visualization, atau Prediction
menu = st.sidebar.selectbox("Menu", ["Overview", "Data Visualization", "Prediction", "About Us"])

# Fungsi untuk memuat dataset dan menyimpannya di cache
@st.cache_data
def load_data():
    # Membaca dataset dari file CSV
    df = pd.read_csv("LaptopPrice.csv")
    return df

# Memuat data ke dalam dataframe
df = load_data()

# Menu: Overview - Menampilkan gambaran umum dataset
if menu == "Overview":
    st.subheader("Gambaran Dataset")
    st.write(f"Total baris dalam dataset: {df.shape[0]}")  # Menampilkan total baris dalam dataset
    st.dataframe(df)  # Menampilkan seluruh dataset dalam format interaktif
    st.write("Shape dari dataset:", df.shape)  # Menampilkan dimensi dataset (baris dan kolom)
    st.write("Informasi Kolom:")
    st.write(df.info())  # Menampilkan informasi tentang kolom dataset (tipe data, non-null values)
    st.write("Nilai yang hilang dalam dataset:")
    st.write(df.isnull().sum())  # Menampilkan jumlah nilai yang hilang (null) pada setiap kolom

# Menu: Data Visualization - Menampilkan visualisasi data
elif menu == "Data Visualization":
    st.subheader("Visualisasi Data")

    # Pilih kolom untuk sumbu X dan Y untuk visualisasi
    x_col = st.selectbox("Pilih kolom X-axis", df.columns)
    y_col = st.selectbox("Pilih kolom Y-axis", df.columns)

    # Pilih jenis plot yang akan ditampilkan
    plot_type = st.selectbox("Pilih jenis plot", ["Barplot", "Lineplot", "Boxplot", "Scatterplot", "Histogram"])

    # Membuat plot berdasarkan pilihan pengguna
    fig, ax = plt.subplots()
    if plot_type == "Barplot":
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat barplot
        explanation = f"Barplot ini menunjukkan hubungan antara {x_col} dan {y_col}. Setiap bar mewakili nilai {x_col}, dan tinggi bar menggambarkan nilai rata-rata {y_col}."
        further_analysis = f"Melihat barplot ini, kita dapat menganalisis apakah terdapat hubungan yang kuat antara {x_col} dan {y_col}. Jika bar lebih tinggi pada nilai tertentu, ini menandakan pengaruh yang lebih besar pada {y_col}."
    elif plot_type == "Lineplot":
        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat lineplot
        explanation = f"Lineplot ini menggambarkan tren {y_col} terhadap {x_col}. Garis yang terbentuk menunjukkan perubahan nilai {y_col} seiring dengan perubahan {x_col}."
        further_analysis = f"Dari lineplot ini, kita dapat melihat apakah {y_col} memiliki pola naik/turun yang konsisten seiring dengan perubahan {x_col}. Jika garis menunjukkan tren yang jelas, ini bisa menjadi indikator penting."
    elif plot_type == "Boxplot":
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat boxplot
        explanation = f"Boxplot ini digunakan untuk menunjukkan distribusi {y_col} untuk setiap kategori {x_col}. Boxplot membantu untuk mengidentifikasi outlier dan persebaran data."
        further_analysis = f"Boxplot ini memberi gambaran distribusi dan keberadaan outlier pada {y_col}. Jika ada nilai yang sangat jauh dari box, itu bisa menjadi indikasi data yang tidak biasa (outlier)."
    elif plot_type == "Scatterplot":
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)  # Membuat scatterplot
        explanation = f"Scatterplot ini menunjukkan hubungan antara {x_col} dan {y_col}. Setiap titik mewakili pasangan nilai dari kedua variabel tersebut."
        further_analysis = f"Dari scatterplot ini, kita bisa melihat pola hubungan antara {x_col} dan {y_col}. Jika titik-titik membentuk pola linier, ini menandakan hubungan yang kuat antara kedua variabel."
    elif plot_type == "Histogram":
        sns.histplot(data=df, x=x_col, ax=ax)  # Membuat histogram
        explanation = f"Histogram ini menggambarkan distribusi frekuensi nilai {x_col}. Sumbu X menunjukkan rentang nilai, dan sumbu Y menunjukkan jumlah data yang jatuh dalam rentang tersebut."
        further_analysis = f"Histogram ini menunjukkan distribusi dari {x_col}. Jika bentuknya cenderung miring atau tidak simetris, ini bisa memberi petunjuk tentang distribusi data yang tidak normal."

    # Menampilkan plot yang telah dibuat
    st.pyplot(fig)

    # Menampilkan penjelasan untuk grafik yang dipilih
    st.write(explanation)
    st.write(further_analysis)  # Menambahkan penjelasan lebih lanjut mengenai analisis grafik


# Menu: Prediction - Menampilkan fitur untuk prediksi harga laptop
elif menu == "Prediction":
    st.subheader("Prediksi Harga Laptop")

    # Mendefinisikan kolom numerik yang akan digunakan sebagai fitur
    numeric_cols = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']
    target_col = 'Price'  # Kolom target untuk prediksi harga laptop

    # Mengkodekan kolom kategorikal (jika ada)
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()  # Membuat objek LabelEncoder
            df[column] = le.fit_transform(df[column])  # Mengkodekan data kategorikal menjadi angka

    # Memisahkan fitur dan target
    X = df[numeric_cols]  # Fitur (kolom numerik)
    y = df[target_col]  # Target (harga laptop)

    # Membagi data menjadi data latih (train) dan data uji (test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melakukan skala fitur agar memiliki distribusi yang seragam
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Melatih dan menerapkan skala pada data latih
    X_test_scaled = scaler.transform(X_test)  # Menerapkan skala pada data uji

    # Melatih model regresi linear
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)  # Melatih model dengan data latih yang telah diskalakan

    # Simpan model dan scaler hanya sekali (jika belum ada)
    if 'laptop_price_model.sav' not in st.session_state:
        joblib.dump(model, "laptop_price_model.sav")  # Menyimpan model ke file
        joblib.dump(scaler, "scaler.sav")  # Menyimpan scaler ke file
        st.session_state['laptop_price_model.sav'] = True  # Tandai bahwa model sudah disimpan
        st.session_state['scaler.sav'] = True  # Tandai bahwa scaler sudah disimpan
        st.write("Model dan scaler disimpan!")  # Memberikan notifikasi bahwa model dan scaler telah disimpan

    # Input pengguna untuk spesifikasi laptop yang akan diprediksi
    st.write("Masukkan spesifikasi laptop:")
    processor_speed = st.number_input("Kecepatan Prosesor (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    ram_size = st.number_input("Ukuran RAM (GB)", min_value=2, max_value=64, value=8, step=2)
    storage_capacity = st.number_input("Kapasitas Penyimpanan (GB)", min_value=128, max_value=2048, value=512, step=128)
    screen_size = st.number_input("Ukuran Layar (inci)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    weight = st.number_input("Berat (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    # Memuat model dan scaler yang sudah disimpan
    model = joblib.load("laptop_price_model.sav")
    scaler = joblib.load("scaler.sav")

    # Prediksi harga laptop berdasarkan input pengguna
    if st.button("Prediksi Harga"):
        user_input = scaler.transform([[processor_speed, ram_size, storage_capacity, screen_size, weight]])  # Menyusun input pengguna dan menormalisasinya
        predicted_price = model.predict(user_input)  # Melakukan prediksi harga
        st.write(f"Prediksi Harga Laptop: ${predicted_price[0]:,.2f}")  # Menampilkan harga yang diprediksi

    # Menampilkan kinerja model (R-squared)
    r_squared = model.score(X_test_scaled, y_test)  # Menghitung R-squared model
    st.write(f"R-squared Model: {r_squared:.2f}")  # Menampilkan nilai R-squared

elif menu == "About Us":
    st.title("About Us")
    st.write("""
    Selamat datang di aplikasi Machine Learning LaptopPrice!

    Aplikasi ini dirancang untuk membantu pengguna memprediksi harga laptop berdasarkan berbagai fitur menggunakan algoritma machine learning. Tujuan kami adalah untuk memberikan prediksi harga yang akurat dan bermanfaat, agar pengguna dapat membuat keputusan pembelian yang lebih bijak.

    **Fitur-fitur aplikasi kami meliputi:**
    - **Pemodelan Prediktif:** Menggunakan teknik machine learning canggih untuk memprediksi harga laptop.
    - **Antarmuka Ramah Pengguna:** Antarmuka yang mudah dinavigasi sehingga memudahkan prediksi harga.
    - **Visualisasi Data:** Grafik dan bagan yang mendetail untuk membantu pengguna memahami tren dan pola harga.

    **Tentang Tim Kami:**
    Tim kami terdiri dari ilmuwan data dan insinyur perangkat lunak yang berdedikasi untuk membuat teknologi yang bermanfaat dan mudah diakses untuk semua orang. Kami percaya pada kekuatan machine learning untuk mengubah cara kita membuat keputusan dan berharap aplikasi ini dapat memberikan nilai tambah pada pengalaman pembelian laptop Anda.

    Terima kasih telah menggunakan aplikasi kami!

    **Informasi Kontak:**
    - Email: support@laptoppriceapp.com
    - Situs Web: www.laptoppriceapp.com
    """)