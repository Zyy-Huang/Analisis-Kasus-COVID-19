import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import chardet
import joblib

st.set_page_config(
    page_title='Analisis Kasus COVID-19',
    page_icon=':earth_americas:',
)

# Fungsi untuk mendeteksi encoding file CSV jika tersedia
def detect_encoding(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read(100000))
            return result["encoding"]
    return "utf-8"  # Gunakan default jika file tidak ditemukan

# Fungsi Gompertz untuk prediksi
def gompertz(a, c, t, t_0):
    Q = a * np.exp(-np.exp(-c * (t - t_0)))
    return Q

# Fungsi utama untuk aplikasi Streamlit
def main():
    # Judul aplikasi 
    st.title(":syringe: :Green[Analisis Kasus COVID-19]")
    st.title("Kelompok-6:")
    st.markdown("Mohammad Ilham Fauzy")
    st.markdown("Nabila Zahra Alia")
    st.markdown("Muhammad Amir Dzakwan")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    st.markdown("Aplikasi ini menampilkan data kasus COVID-19 berdasarkan analisis yang mendalam.")
    # Cek file yang tersedia dalam direktori saat ini dan dalam folder dataset
    available_files = os.listdir(".")
    dataset_files = os.listdir("dataset") if os.path.exists("dataset") else []
    st.sidebar.write("Files in current directory:", available_files)
    st.sidebar.write("Files in 'dataset/' folder:", dataset_files)

    # Tentukan path file berdasarkan lokasi dalam folder dataset
    patient_path = "dataset/patient.csv" if "patient.csv" in dataset_files else None
    confirmed_path = "dataset/confirmed_acc.csv" if "confirmed_acc.csv" in dataset_files else None
    province_path = "dataset/province.csv" if "province.csv" in dataset_files else None

    # Load dataset hanya jika file tersedia
    if patient_path:
        df_patient = pd.read_csv(patient_path, encoding=detect_encoding(patient_path))
    else:
        st.error("File 'patient.csv' tidak ditemukan dalam folder dataset!")

    if confirmed_path:
        df_confirmed = pd.read_csv(confirmed_path, encoding=detect_encoding(confirmed_path))
    else:
        st.error("File 'confirmed_acc.csv' tidak ditemukan dalam folder dataset!")

    if province_path:
        df_province = pd.read_csv(province_path, encoding=detect_encoding(province_path))
    else:
        st.error("File 'province.csv' tidak ditemukan dalam folder dataset!")

    # Lanjutkan hanya jika semua file ditemukan
    if patient_path and confirmed_path and province_path:
        # 1. Sebelum Penyebaran Kasus COVID
        st.subheader("Sebelum Penyebaran Kasus COVID")
        pre_covid_data = df_confirmed.head(5)  # Ambil 5 data pertama
        st.table(pre_covid_data)

        # 2. Kasus Pertama Kali Muncul dari Dataset (5 Data Berurutan)
        st.subheader("Kasus Pertama Kali Muncul")
        df_confirmed["date"] = pd.to_datetime(df_confirmed["date"])  # Pastikan kolom tanggal dalam format datetime
        first_case_index = df_confirmed[df_confirmed["cases"] > 0].index.min()  # Temukan indeks kasus pertama yang muncul

        if first_case_index is not None:
            first_case_dates = df_confirmed.loc[first_case_index:first_case_index + 4]  # Ambil 5 data berurutan
            first_case_dates["date"] = first_case_dates["date"].dt.strftime('%d/%m/%Y')  # Format tanggal
            st.table(first_case_dates[["date", "cases"]])  # Tampilkan tanggal dan jumlah kasus
        else:
            st.warning("Tidak ada kasus yang terdeteksi dalam dataset.")

        # 3. Tanggal kemungkinan kasus pertama terjadi
        if first_case_index is not None:
            first_case_date = df_confirmed.loc[first_case_index, "date"].strftime('%d/%m/%Y')
            st.write(f"Kasus pertama kemungkinan terjadi pada {first_case_date}.")
        else:
            st.warning("Tidak ada kasus yang terdeteksi dalam dataset.")

        # 4. Tren Kasus dari Waktu ke Waktu
        st.subheader("Tren Kasus dari Waktu ke Waktu")
        fig_trend = px.line(df_confirmed, x="date", y="cases", title="Tren Kasus COVID-19 dari Waktu ke Waktu")
        st.plotly_chart(fig_trend)

        # 5. Peningkatan Kasus dari Awal Maret
        st.subheader("Peningkatan Kasus dari Awal Maret")
        march_data = df_confirmed[df_confirmed["date"] >= "2020-03-01"]
        st.line_chart(march_data.set_index("date")["cases"])

        # 6. Peningkatan Kasus yang Terjadi dari Awal Maret (dalam bentuk tabel)
        st.subheader("Peningkatan Kasus yang Terjadi dari Awal Maret")
        st.table(march_data)

        # 7. Prediksi Kasus untuk 60 Hari ke Depan menggunakan model Gompertz
        st.subheader("Prediksi Kasus untuk 60 Hari ke Depan")
        df_confirmed['days'] = (df_confirmed['date'] - df_confirmed['date'].min()).dt.days
        x = list(df_confirmed['days'])
        y = list(df_confirmed['cases'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.1, shuffle=False)
        x_test_added = x_test + list(range(max(x_test) + 1, max(x_test) + 61))  # Tambahkan 60 hari ke x_test

        # Fit model Gompertz
        popt, pcov = curve_fit(gompertz, x_train, y_train, method='trf', bounds=([100, 0, 0], [6 * max(y_train), 0.15, 70]))
        a, estimated_c, estimated_t_0 = popt
        y_pred = gompertz(a, estimated_c, x_train + x_test_added, estimated_t_0)

        # Menampilkan hasil prediksi
        prediction_df = pd.DataFrame({"days": x_train + x_test_added, "predicted_cases": y_pred})
        fig_prediction = go.Figure()
        fig_prediction.add_trace(go.Scatter(x=prediction_df['days'], y=prediction_df['predicted_cases'], mode='lines', name='Prediksi Kasus'))
        fig_prediction.update_layout(title='Hasil Prediksi Kasus COVID-19 untuk 60 Hari ke Depan',
                                      xaxis_title='Hari Sejak 1 Maret 2020',
                                      yaxis_title='Kasus Positif Terkonfirmasi')
        st.plotly_chart(fig_prediction)

        # 8. Gambaran Grafik Prediksi vs Data Nyata
        st.subheader("Gambaran Grafik Prediksi vs Data Nyata")
        actual_cases = march_data.set_index("date")["cases"].reindex(pd.date_range(start="2020-03-01", periods=len(march_data), freq='D')).fillna(0).values
        days_actual = list(range(len(actual_cases)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days_actual, y=actual_cases, mode='lines', name='Data Terkonfirmasi', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=prediction_df['days'], y=prediction_df['predicted_cases'], mode='lines', name='Prediksi Data'))
        fig.update_layout(title='Prediksi vs Data Terkonfirmasi Kasus COVID-19 di Indonesia',
                          xaxis_title='Hari Sejak 1 Maret 2020',
                          yaxis_title='Kasus Positif Terkonfirmasi')
        st.plotly_chart(fig)

        # 9. Hasil Prediksi COVID-19 untuk 60 Hari
        st.subheader("Hasil Prediksi COVID-19 untuk 60 Hari")
        st.dataframe(prediction_df)

        # 10. Perbandingan akurasi model 
        X = df_patient.drop(columns=['current_state', 'confirmed_date', 'released_date', 'deceased_date'])
        y = df_patient['current_state'].apply(lambda x: 1 if x == 'released' else 0)

        # Tanpa Data Preparation
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X.select_dtypes(include='number'), y, test_size=0.2, random_state=42)
        model_1 = RandomForestClassifier(random_state=42)
        model_1.fit(X_train_1, y_train_1)
        y_pred_1 = model_1.predict(X_test_1)
        accuracy_1 = accuracy_score(y_test_1, y_pred_1)

        # Dengan Data Preparation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include='number'))
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model_2 = RandomForestClassifier(random_state=42)
        model_2.fit(X_train_2, y_train_2)
        y_pred_2 = model_2.predict(X_test_2)
        accuracy_2 = accuracy_score(y_test_2, y_pred_2)

        # Menampilkan hasil akurasi
        st.write("### Perbandingan Akurasi Model")
        st.write(f"**Akurasi Model Tanpa Persiapan Data:** {accuracy_1:.2f}")
        st.write(f"**Akurasi Model Dengan Persiapan Data:** {accuracy_2:.2f}")
        # Menyimpan model terbaik
        best_model = model_2 if accuracy_2 > accuracy_1 else model_1
        joblib.dump(best_model, 'best_model.pkl')

        # 11. Informasi Pasien
        st.subheader("Informasi Pasien")
        df_patient_filtered = df_patient.dropna(subset=["gender", "age", "nationality", "province"])
        st.dataframe(df_patient_filtered)

        # 12. Masukkan Data Pasien
        st.subheader("Masukkan Data Pasien")
        age = st.number_input("Umur Pasien", min_value=0, max_value=120, step=1)
        gender_options = df_patient["gender"].unique() if "gender" in df_patient.columns else ["Laki-laki", "Perempuan"]
        gender = st.selectbox("Jenis Kelamin", gender_options)
        region_options = df_patient["province"].unique() if "province" in df_patient.columns else ["DKI Jakarta"]
        region = st.selectbox("Wilayah", region_options)

        matching_patient = df_patient[
            (df_patient["age"] == age) & 
            (df_patient["gender"] == gender) & 
            (df_patient["province"] == region)
        ]

        if st.button("Status Pasien"):
            if not matching_patient.empty:
                actual_status = matching_patient.iloc[0]["current_state"]
                if actual_status == "released":
                    status = "Sembuh"
                elif actual_status == "deceased":
                    status = "Meninggal"
                else:
                    status = "Masih Dirawat"
                st.success(f"Status: {status} (Sesuai Data Historis)")
            else:
                st.warning("Data pasien tidak ditemukan dalam dataset.")

        # 13. Data Pasien Berdasarkan Status
        st.subheader("Data Pasien Berdasarkan Status")
        patient_status_counts = df_patient["current_state"].value_counts()
        st.bar_chart(patient_status_counts)

        # 14. Rata-Rata Usia Berdasarkan Gender
        st.subheader("Rata-Rata Usia Berdasarkan Gender")
        average_age = df_patient.groupby("gender")["age"].mean()
        st.write(average_age)

        # 15. Grafik Kasus Berdasarkan Gender
        st.subheader("Grafik Kasus Berdasarkan Gender")
        gender_status_counts = df_patient.groupby(["gender", "current_state"]).size().unstack()
        fig_gender = px.bar(
            gender_status_counts,
            title="Kasus Berdasarkan Gender dan Status",
            labels={"value": "Jumlah Kasus", "gender": "Gender"},
            barmode="group",
        )
        st.plotly_chart(fig_gender)

        # 16. Grafik Daerah dengan Kasus Terbanyak
        st.subheader("Grafik Daerah dengan Kasus Terbanyak")
        region_counts = df_patient["province"].value_counts()
        st.bar_chart(region_counts)

        # 17. Grafik Pasien yang Positif Berdasarkan Tanggal Terkonfirmasinya
        st.subheader("Grafik Pasien yang Positif Berdasarkan Tanggal Terkonfirmasinya")
        st.line_chart(df_confirmed.set_index("date")["cases"])

        # 18. Data Kasus COVID-19 berdasarkan Pulau
        st.subheader("Data Kasus COVID-19 berdasarkan Pulau")
        island_data = df_province.groupby('island')['confirmed'].sum().reset_index()  # Mengambil data dari province.csv
        st.dataframe(island_data)

        # 19. Grafik Distribusi Kasus COVID-19
        st.subheader("Grafik Distribusi Kasus COVID-19")
        fig_pie = px.pie(
            data_frame=island_data,
            names="island",
            values="confirmed",
            title="Distribusi Kasus Terkonfirmasi Berdasarkan Pulau",
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        st.plotly_chart(fig_pie)

        # 20. Analisis Data
        st.subheader("Analisis Data")
        max_cases = island_data.loc[island_data['confirmed'].idxmax()]
        min_cases = island_data.loc[island_data['confirmed'].idxmin()]
        st.write(f"Pulau dengan jumlah kasus terkonfirmasi tertinggi: {max_cases['island']} ({max_cases['confirmed']})")
        st.write(f"Pulau dengan jumlah kasus terkonfirmasi terendah: {min_cases['island']} ({min_cases['confirmed']})")
        st.write(f"Rata-rata jumlah kasus terkonfirmasi: {island_data['confirmed'].mean():.2f}")
        st.write(f"Total jumlah kasus terkonfirmasi: {island_data['confirmed'].sum()}")

if __name__ == "__main__":
    main()
