
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# PROYEK STATISTIKA KELOMPOK 15
# ==========================================

# 1. MEMBACA DATASET
df = pd.read_csv("dataset/Kelompok_15_Tema_B_Kualitas_Udara.csv")

os.makedirs("hasil", exist_ok=True)

print("=" * 55)
print("ANALISIS STATISTIKA KELOMPOK 15")
print("=" * 55)

print("\nJUMLAH DATA:", df.shape)
print("\n5 DATA PERTAMA:")
print(df.head())


skala = {
    "ID_Observasi": "Nominal (identitas)",
    "Zona_Lokasi": "Nominal",
    "Jenis_Hari": "Nominal",
    "Jam_Sibuk": "Nominal",
    "Volume_Kendaraan": "Rasio",
    "Kecepatan_Angin_KmJ": "Rasio",
    "Suhu_Celcius": "Interval",
    "Indeks_PM25": "Rasio",
    "Status_Udara": "Ordinal"
}

tabel_skala = pd.DataFrame(
    list(skala.items()),
    columns=["Variabel", "Skala Pengukuran"]
)

print("\nSKALA PENGUKURAN:")
print(tabel_skala.to_string(index=False))

tabel_skala.to_csv("hasil/skala_pengukuran.csv", index=False)

# ==========================================
# 3. PEMERIKSAAN MISSING VALUES
# ==========================================

print("\nMISSING VALUES SEBELUM PEMBERSIHAN:")
print(df.isnull().sum())

missing = df.isnull().sum().reset_index()
missing.columns = ["Variabel", "Jumlah Missing"]

missing.to_csv("hasil/missing_values.csv", index=False)

# Salin data asli
df_bersih = df.copy()

# Mengisi missing values dengan median
kolom_numerik = [
    "Volume_Kendaraan",
    "Kecepatan_Angin_KmJ",
    "Suhu_Celcius",
    "Indeks_PM25"
]

for kolom in kolom_numerik:
    median = df_bersih[kolom].median()
    df_bersih[kolom] = df_bersih[kolom].fillna(median)

print("\nMISSING VALUES SETELAH IMPUTASI:")
print(df_bersih.isnull().sum())

# ==========================================
# 4. DETEKSI OUTLIER DENGAN METODE IQR
# ==========================================

print("\nHASIL DETEKSI OUTLIER:")

hasil_iqr = []

for kolom in kolom_numerik:

    Q1 = df_bersih[kolom].quantile(0.25)
    Q3 = df_bersih[kolom].quantile(0.75)

    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    jumlah_outlier = (
        (df_bersih[kolom] < batas_bawah) |
        (df_bersih[kolom] > batas_atas)
    ).sum()

    print(f"\nVariabel: {kolom}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Batas bawah:", batas_bawah)
    print("Batas atas:", batas_atas)
    print("Jumlah outlier:", jumlah_outlier)

    hasil_iqr.append({
        "Variabel": kolom,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Batas Bawah": batas_bawah,
        "Batas Atas": batas_atas,
        "Jumlah Outlier": jumlah_outlier
    })

    # Penanganan outlier dengan capping
    df_bersih[kolom] = df_bersih[kolom].clip(
        lower=batas_bawah,
        upper=batas_atas
    )

tabel_iqr = pd.DataFrame(hasil_iqr)
tabel_iqr.to_csv("hasil/deteksi_outlier.csv", index=False)

# ==========================================
# 5. BOXPLOT SEBELUM DAN SESUDAH PENANGANAN
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, kolom in enumerate(kolom_numerik):

    sns.boxplot(
        y=df[kolom],
        ax=axes[i // 2, i % 2]
    )

    axes[i // 2, i % 2].set_title(
        f"Sebelum Penanganan: {kolom}"
    )

plt.tight_layout()
plt.savefig("hasil/boxplot_sebelum.png", dpi=300)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, kolom in enumerate(kolom_numerik):

    sns.boxplot(
        y=df_bersih[kolom],
        ax=axes[i // 2, i % 2]
    )

    axes[i // 2, i % 2].set_title(
        f"Sesudah Penanganan: {kolom}"
    )

plt.tight_layout()
plt.savefig("hasil/boxplot_sesudah.png", dpi=300)
plt.show()

# ==========================================
# 6. STATISTIKA DESKRIPTIF
# ==========================================

hasil_statistik = []

for kolom in kolom_numerik:

    data = df_bersih[kolom]

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    hasil_statistik.append({
        "Variabel": kolom,
        "Mean": data.mean(),
        "Median": data.median(),
        "Modus": data.mode().iloc[0],
        "Standar Deviasi": data.std(),
        "Varians": data.var(),
        "IQR": Q3 - Q1,
        "Skewness": data.skew()
    })

tabel_statistik = pd.DataFrame(hasil_statistik)

print("\nTABEL STATISTIKA DESKRIPTIF:")
print(tabel_statistik.round(3).to_string(index=False))

tabel_statistik.to_csv(
    "hasil/statistika_deskriptif.csv",
    index=False
)

# ==========================================
# 7. GRAFIK HISTOGRAM
# ==========================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df_bersih,
    x="Indeks_PM25",
    bins=20,
    kde=True
)

plt.title("Distribusi Indeks PM2.5")
plt.xlabel("Indeks PM2.5")
plt.ylabel("Frekuensi")
plt.tight_layout()

plt.savefig("hasil/histogram_pm25.png", dpi=300)
plt.show()

# ==========================================
# 8. GROUPED BAR CHART
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_bersih,
    x="Zona_Lokasi",
    y="Indeks_PM25",
    hue="Jenis_Hari",
    errorbar=None
)

plt.title("Rata-rata Indeks PM2.5 Berdasarkan Zona dan Jenis Hari")
plt.xlabel("Zona Lokasi")
plt.ylabel("Rata-rata Indeks PM2.5")
plt.xticks(rotation=15)
plt.legend(title="Jenis Hari")
plt.tight_layout()

plt.savefig("hasil/grouped_bar_chart.png", dpi=300)
plt.show()

# ==========================================
# 9. MENYIMPAN DATASET BERSIH
# ==========================================

df_bersih.to_csv(
    "hasil/dataset_bersih.csv",
    index=False
)

print("\n" + "=" * 55)
print("ANALISIS MINGGU 1 SELESAI!")
print("Semua hasil tersimpan di folder hasil.")
print("=" * 55)