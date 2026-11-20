# ============================================================
# PROGRESS 2 - ANALISIS PROBABILITAS
# KELOMPOK 15 | TEMA B: KUALITAS UDARA
# STATISTIKA DAN PROBABILITAS
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, binom
import os

# 1. MEMUAT DATA
df = pd.read_csv("hasil/dataset_bersih.csv")

os.makedirs("hasil", exist_ok=True)

print("=" * 60)
print("ANALISIS PROBABILITAS - KUALITAS UDARA")
print("=" * 60)


# ------------------------------------------------------------
# 2. PROBABILITAS KONDISI UDARA
# ------------------------------------------------------------

print("\n1. PROBABILITAS KONDISI UDARA")

frekuensi = df["Status_Udara"].value_counts()
probabilitas = df["Status_Udara"].value_counts(normalize=True)

hasil_kondisi = pd.DataFrame({
    "Frekuensi": frekuensi,
    "Probabilitas": probabilitas
})

print(hasil_kondisi)

hasil_kondisi.to_csv(
    "hasil/probabilitas_kondisi_udara.csv"
)


# ------------------------------------------------------------
# 3. DISTRIBUSI NORMAL PM2.5
# ------------------------------------------------------------

print("\n2. DISTRIBUSI NORMAL PM2.5")

data_pm25 = df["Indeks_PM25"]

mean_pm25 = data_pm25.mean()
std_pm25 = data_pm25.std()

# Menghitung peluang PM2.5 melebihi nilai rata-rata
# ditambah satu standar deviasi
batas = mean_pm25 + std_pm25

peluang_lebih = 1 - norm.cdf(
    batas, mean_pm25, std_pm25
)

print("Mean PM2.5:", mean_pm25)
print("Standar deviasi:", std_pm25)
print("Batas nilai:", batas)
print("Peluang PM2.5 melebihi batas:", peluang_lebih)

hasil_normal = pd.DataFrame({
    "Mean": [mean_pm25],
    "Standar_Deviasi": [std_pm25],
    "Batas": [batas],
    "Peluang_Melebihi_Batas": [peluang_lebih]
})

hasil_normal.to_csv(
    "hasil/distribusi_normal_pm25.csv",
    index=False
)


# ------------------------------------------------------------
# 4. DISTRIBUSI BINOMIAL
# ------------------------------------------------------------

print("\n3. DISTRIBUSI BINOMIAL")

# Menentukan kejadian PM2.5 di atas rata-rata
# sebagai kejadian yang diteliti

kejadian = (df["Indeks_PM25"] > mean_pm25)

p = kejadian.mean()

n = 10
k = 5

peluang_binomial = binom.pmf(k, n, p)

print("Probabilitas kejadian:", p)
print("Jumlah percobaan:", n)
print("Jumlah kejadian yang diharapkan:", k)
print("Peluang tepat 5 kejadian:", peluang_binomial)

hasil_binomial = pd.DataFrame({
    "n": [n],
    "k": [k],
    "p": [p],
    "Peluang": [peluang_binomial]
})

hasil_binomial.to_csv(
    "hasil/distribusi_binomial.csv",
    index=False
)


# ------------------------------------------------------------
# 5. PROBABILITAS BERSYARAT
# ------------------------------------------------------------

print("\n4. PROBABILITAS BERSYARAT")

# Peluang PM2.5 di atas rata-rata pada Pusat Kota
# dibandingkan dengan seluruh data Pusat Kota

pusat_kota = df[df["Zona_Lokasi"] == "Pusat Kota"]

if len(pusat_kota) > 0:

    peluang_bersyarat = (
        pusat_kota["Indeks_PM25"] > mean_pm25
    ).mean()

    print(
        "Peluang PM2.5 di atas rata-rata "
        "pada Pusat Kota:",
        peluang_bersyarat
    )

    hasil_bersyarat = pd.DataFrame({
        "Zona": ["Pusat Kota"],
        "Peluang": [peluang_bersyarat]
    })

    hasil_bersyarat.to_csv(
        "hasil/probabilitas_bersyarat.csv",
        index=False
    )

else:
    print("Data Pusat Kota tidak ditemukan.")


# ------------------------------------------------------------
# 6. SELESAI
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ANALISIS PROGRESS 2 SELESAI!")
print("Hasil tersimpan di folder hasil.")
print("=" * 60)