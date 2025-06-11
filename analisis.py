import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
from io import BytesIO
from datetime import datetime


# Konfigurasi halaman
st.set_page_config(page_title="Analisis Pasar Sumedang", layout="wide")

# Navigasi Halaman
page = st.sidebar.radio("Navigasi", ["📊 EDA", "📈 K-Means Clustering"])

# Fungsi untuk memuat data
@st.cache_data

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df[df['satuan_item'] == 'kg']
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['tahun'] = df['tanggal'].dt.year
    df['bulan'] = df['tanggal'].dt.month
    df = df.dropna(subset=['jumlah', 'kebutuhan'])
    df['jumlah'] = df['jumlah'].astype(int)
    df['kebutuhan'] = df['kebutuhan'].astype(int)
    return df

# Halaman EDA
if page == "📊 EDA":
    st.title("📊 Dashboard Analisis Barang di Pasar Tradisional Sumedang")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        
        # Filter Sidebar
        st.sidebar.header("Filter")
        pasar_filter = st.sidebar.multiselect("Pilih Pasar:", df['nama_pasar'].unique(), default=df['nama_pasar'].unique())
        barang_filter = st.sidebar.multiselect("Pilih Barang:", df['item_barang'].unique(), default=df['item_barang'].unique())
        tahun_filter = st.sidebar.multiselect("Pilih Tahun:", sorted(df['tahun'].unique()), default=sorted(df['tahun'].unique()))
        bulan_filter = st.sidebar.multiselect("Pilih Bulan:", sorted(df['bulan'].unique()), default=sorted(df['bulan'].unique()))

        # Terapkan Filter
        filtered_df = df[
            (df['nama_pasar'].isin(pasar_filter)) &
            (df['item_barang'].isin(barang_filter)) &
            (df['tahun'].isin(tahun_filter)) &
            (df['bulan'].isin(bulan_filter))
        ]

        # Chart 1
        st.subheader("Distribusi Jumlah dan Kebutuhan Barang")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['jumlah'], bins=50, color='blue', label='Jumlah', kde=True, ax=ax1)
        sns.histplot(filtered_df['kebutuhan'], bins=50, color='orange', label='Kebutuhan', kde=True, ax=ax1)
        ax1.set_xlabel("Jumlah / Kebutuhan")
        ax1.set_ylabel("Frekuensi")
        ax1.legend()
        st.pyplot(fig1)

        avg_jumlah = filtered_df['jumlah'].mean()
        avg_kebutuhan = filtered_df['kebutuhan'].mean()
        median_jumlah = filtered_df['jumlah'].median()
        median_kebutuhan = filtered_df['kebutuhan'].median()
        st.markdown("### 🔍 Insight Distribusi")
        st.markdown(f"- Rata-rata stok: **{avg_jumlah:.0f}**, Median: **{median_jumlah:.0f}**")
        st.markdown(f"- Rata-rata kebutuhan: **{avg_kebutuhan:.0f}**, Median: **{median_kebutuhan:.0f}**")

        if avg_jumlah > avg_kebutuhan:
            st.info("📦 Rata-rata stok lebih tinggi dari kebutuhan. Potensi overstock.")
        elif avg_jumlah < avg_kebutuhan:
            st.warning("⚠️ Rata-rata kebutuhan lebih tinggi dari stok. Potensi kekurangan barang.")
        else:
            st.success("✅ Stok dan kebutuhan seimbang.")

        # Chart 2
        st.subheader("Hubungan Jumlah vs Kebutuhan")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=filtered_df, x='jumlah', y='kebutuhan', alpha=0.4, ax=ax2)
        st.pyplot(fig2)

        correlation = filtered_df['jumlah'].corr(filtered_df['kebutuhan'])
        st.markdown("### 🔍 Insight Korelasi")
        st.markdown(f"- Korelasi: **{correlation:.2f}**")
        if correlation > 0.7:
            st.success("📈 Hubungan kuat: stok cenderung sesuai kebutuhan.")
        elif correlation > 0.4:
            st.info("📉 Korelasi sedang.")
        else:
            st.warning("❗ Korelasi lemah: stok tidak berbasis permintaan.")

        # Chart 3
        st.subheader("Barang dengan Kebutuhan Tertinggi")
        top_items = filtered_df.groupby('item_barang')[['jumlah', 'kebutuhan']].mean().sort_values(by='kebutuhan', ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(10,6))
        top_items.plot(kind='bar', ax=ax3, color=['skyblue', 'salmon'])
        st.pyplot(fig3)

        most_needed = top_items.index[0]
        second_needed = top_items.index[1]
        st.markdown("### 🔍 Insight Kebutuhan Barang")
        st.markdown(f"- Paling dibutuhkan: **{most_needed}**, disusul **{second_needed}**")

        # Chart 4
        st.subheader("Rata-rata Jumlah dan Kebutuhan per Pasar")
        pasar_stats = filtered_df.groupby('nama_pasar')[['jumlah', 'kebutuhan']].mean().sort_values(by='kebutuhan', ascending=False).head(5)
        fig4, ax4 = plt.subplots(figsize=(10,6))
        pasar_stats.plot(kind='barh', ax=ax4, color=['mediumseagreen', 'tomato'])
        st.pyplot(fig4)

        st.markdown("### 🔍 Insight Pasar")
        st.markdown(f"- Kebutuhan tertinggi: **{pasar_stats.index[0]}**")
        st.markdown(f"- Terendah di antara top 5: **{pasar_stats.index[-1]}**")

# Halaman K-Means Clustering
elif page == "📈 K-Means Clustering":
    st.title("📈 K-Means Clustering: Pengelompokan Barang Berdasarkan Jumlah & Kebutuhan")

    uploaded_file = st.file_uploader("Unggah dataset", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)

        # Sidebar Filter
        st.sidebar.header("Filter")
        pasar_filter = st.sidebar.multiselect("Pilih Pasar:", df['nama_pasar'].unique(), default=df['nama_pasar'].unique())
        tahun_filter = st.sidebar.multiselect("Pilih Tahun:", sorted(df['tahun'].unique()), default=sorted(df['tahun'].unique()))
        bulan_filter = st.sidebar.multiselect("Pilih Bulan:", sorted(df['bulan'].unique()), default=sorted(df['bulan'].unique()))

        # Terapkan filter
        filtered_df = df[
            (df['nama_pasar'].isin(pasar_filter)) &
            (df['tahun'].isin(tahun_filter)) &
            (df['bulan'].isin(bulan_filter))
        ].copy()

        if filtered_df.empty:
            st.warning("⚠️ Tidak ada data yang cocok dengan filter. Silakan ubah filter Anda.")
        else:
            # Standarisasi untuk clustering
            X = filtered_df[['jumlah', 'kebutuhan']]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k = st.slider("Pilih jumlah cluster (K):", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)

            # Tambahkan hasil cluster ke data asli (gunakan salinan)
            filtered_df['cluster'] = clusters

            # Visualisasi Cluster
            st.subheader("Visualisasi Cluster")
            fig = px.scatter(filtered_df, x='jumlah', y='kebutuhan', color=filtered_df['cluster'].astype(str),
                             title='Hasil Clustering Jumlah vs Kebutuhan',
                             labels={'jumlah': 'Jumlah Barang', 'kebutuhan': 'Kebutuhan Barang', 'cluster': 'Cluster'})
            st.plotly_chart(fig, use_container_width=True)

            # Insight Otomatis
            st.subheader("📌 Insight Otomatis dari Clustering")
            for c in sorted(filtered_df['cluster'].unique()):
                subset = filtered_df[filtered_df['cluster'] == c]
                avg_jumlah = int(subset['jumlah'].mean())
                avg_kebutuhan = int(subset['kebutuhan'].mean())
                top_items = subset['item_barang'].value_counts().head(3).index.tolist()

                st.markdown(f"""
                - **Cluster {c}**
                  - Rata-rata Jumlah: **{avg_jumlah}**
                  - Rata-rata Kebutuhan: **{avg_kebutuhan}**
                  - Contoh Barang Teratas: {', '.join(top_items)}
                  - 📌 Interpretasi: Barang dalam cluster ini cenderung memiliki pola **{'tinggi' if avg_kebutuhan > 1000 else 'rendah'}** kebutuhan.
                """)

            # Export PDF
            st.subheader("📤 Export Laporan Clustering ke PDF")

            # Kelas PDF
            class PDF(FPDF):
                def header(self):
                    # Tambahkan logo jika ada
                    self.image("logo.png", 10, 10, 25)  # kiri, atas, ukuran

                    # Judul / kop surat
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 6, "PEMERINTAH KABUPATEN SUMEDANG", ln=True, align='C')
                    self.cell(0, 6, "DINAS KOPERASI, UKM, PERDAGANGAN DAN PERINDUSTRIAN", ln=True, align='C')

                    self.set_font('Arial', '', 10)
                    self.cell(0, 6, "Jl. Raya Sumedang No.123, Sumedang, Jawa Barat", ln=True, align='C')
                    self.cell(0, 6, "Telp: (0261) 123456 | Email: dinaskopdag@sumedang.go.id", ln=True, align='C')

                    # Garis pemisah
                    self.ln(5)
                    self.line(10, self.get_y(), 200, self.get_y())
                    self.ln(10)

                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.cell(0, 10, f"Halaman {self.page_no()} | Dicetak: {datetime.today().strftime('%d-%m-%Y')}", 0, 0, 'C')


            pdf = PDF()
            pdf.add_page()

            # Keterangan Filter
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Keterangan Filter:", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 8, f"Pasar  : {', '.join(pasar_filter)}", ln=True)
            pdf.cell(0, 8, f"Tahun  : {', '.join(map(str, tahun_filter))}", ln=True)
            pdf.cell(0, 8, f"Bulan  : {', '.join(map(str, bulan_filter))}", ln=True)
            pdf.ln(5)

            # Hasil Clustering
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Hasil Clustering & Insight", ln=True)
            pdf.set_font("Arial", '', 11)

            for c in sorted(filtered_df['cluster'].unique()):
                subset = filtered_df[filtered_df['cluster'] == c]
                avg_jumlah = int(subset['jumlah'].mean())
                avg_kebutuhan = int(subset['kebutuhan'].mean())
                top_items = subset['item_barang'].value_counts().head(3).index.tolist()
                insight = "tinggi" if avg_kebutuhan > 1000 else "rendah"

                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 8, f"Cluster {c}", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.cell(0, 8, f"  - Rata-rata Jumlah     : {avg_jumlah}", ln=True)
                pdf.cell(0, 8, f"  - Rata-rata Kebutuhan  : {avg_kebutuhan}", ln=True)
                pdf.cell(0, 8, f"  - Contoh Barang Teratas: {', '.join(top_items)}", ln=True)
                pdf.cell(0, 8, f"  - Interpretasi         : Pola kebutuhan {insight}", ln=True)
                pdf.ln(2)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(3)

            # Simpan PDF ke memori
            pdf_output = BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0)

            st.download_button(
                label="📄 Unduh Laporan PDF",
                data=pdf_output,
                file_name="laporan_clustering_pasar_sumedang.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("Silakan unggah file dataset terlebih dahulu.")