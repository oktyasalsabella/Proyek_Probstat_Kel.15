# ============================================================
# PROGRESS 2 - MODELING PROBABILITY
# Kelompok 15 - Tema B: Kualitas Udara
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import norm, binom, shapiro

# 1. MEMBACA DATASET BERSIH
data = pd.read_csv("hasil/dataset_bersih.csv")

# ============================================================
# 2. PROBABILITAS EMPIRIS
# ============================================================

jumlah_data = len(data)

# Probabilitas 3 kejadian
kejadian = {
    "Udara Baik": (data["Status_Udara"] == "Baik").sum(),
    "Udara Sedang": (data["Status_Udara"] == "Sedang").sum(),
    "Udara Tidak Sehat": (data["Status_Udara"] == "Tidak Sehat").sum()
}

hasil_empiris = []

for nama, jumlah in kejadian.items():
    hasil_empiris.append({
        "Kejadian": nama,
        "Jumlah": jumlah,
        "Total Data": jumlah_data,
        "Probabilitas": jumlah / jumlah_data
    })

# Probabilitas bersyarat 1:
# P(Status Tidak Sehat | Zona Pusat Kota)

pusat_kota = data[data["Zona_Lokasi"] == "Pusat Kota"]

jumlah_pusat_kota = len(pusat_kota)

jumlah_tidak_sehat_pusat = (
    pusat_kota["Status_Udara"] == "Tidak Sehat"
).sum()

p_tidak_sehat_pusat = (
    jumlah_tidak_sehat_pusat / jumlah_pusat_kota
    if jumlah_pusat_kota > 0 else 0
)

hasil_empiris.append({
    "Kejadian": "P(Tidak Sehat | Pusat Kota)",
    "Jumlah": jumlah_tidak_sehat_pusat,
    "Total Data": jumlah_pusat_kota,
    "Probabilitas": p_tidak_sehat_pusat
})

# Probabilitas bersyarat 2:
# P(Jam Sibuk Ya | Hari Kerja)

hari_kerja = data[data["Jenis_Hari"] == "Hari Kerja"]

jumlah_hari_kerja = len(hari_kerja)

jumlah_jam_sibuk = (
    hari_kerja["Jam_Sibuk"] == "Ya"
).sum()

p_jam_sibuk_hari_kerja = (
    jumlah_jam_sibuk / jumlah_hari_kerja
    if jumlah_hari_kerja > 0 else 0
)

hasil_empiris.append({
    "Kejadian": "P(Jam Sibuk Ya | Hari Kerja)",
    "Jumlah": jumlah_jam_sibuk,
    "Total Data": jumlah_hari_kerja,
    "Probabilitas": p_jam_sibuk_hari_kerja
})

df_empiris = pd.DataFrame(hasil_empiris)

df_empiris.to_csv(
    "hasil/probabilitas_empiris.csv",
    index=False
)

print("\n=== PROBABILITAS EMPIRIS ===")
print(df_empiris)


# ============================================================
# 3. DISTRIBUSI NORMAL PM2.5
# ============================================================

pm25 = data["Indeks_PM25"]

mean_pm25 = pm25.mean()
std_pm25 = pm25.std(ddof=1)

# Uji normalitas Shapiro-Wilk
stat_shapiro, p_shapiro = shapiro(pm25)

# Dua skenario batas PM2.5
batas_skenario = [50, 75]

hasil_normal = []

for batas in batas_skenario:
    peluang = norm.sf(
        batas,
        loc=mean_pm25,
        scale=std_pm25
    )

    hasil_normal.append({
        "Mean": mean_pm25,
        "Standar Deviasi": std_pm25,
        "Batas PM2.5": batas,
        "Peluang Melebihi Batas": peluang
    })

df_normal = pd.DataFrame(hasil_normal)

df_normal["Statistik Shapiro"] = stat_shapiro
df_normal["P_Value Shapiro"] = p_shapiro

df_normal.to_csv(
    "hasil/distribusi_normal_pm25.csv",
    index=False
)

print("\n=== DISTRIBUSI NORMAL PM2.5 ===")
print(df_normal)

print("\n=== UJI NORMALITAS SHAPIRO-WILK ===")
print("Statistik:", stat_shapiro)
print("P-value:", p_shapiro)


# ============================================================
# 4. DISTRIBUSI BINOMIAL
# ============================================================

# Variabel biner:
# Sukses = Jam_Sibuk Ya
# Gagal = Jam_Sibuk Tidak

jumlah_sukses = (data["Jam_Sibuk"] == "Ya").sum()

p = jumlah_sukses / jumlah_data

n = 20
k = 10

# Probabilitas tepat k sukses
prob_tepat_k = binom.pmf(k, n, p)

# Probabilitas minimal k sukses
prob_minimal_k = binom.sf(k - 1, n, p)

hasil_binomial = pd.DataFrame([{
    "Variabel": "Jam_Sibuk",
    "Kategori Sukses": "Ya",
    "Proporsi Sukses (p)": p,
    "n": n,
    "k": k,
    "P(X=k)": prob_tepat_k,
    "P(X>=k)": prob_minimal_k
}])

hasil_binomial.to_csv(
    "hasil/distribusi_binomial.csv",
    index=False
)

print("\n=== DISTRIBUSI BINOMIAL ===")
print(hasil_binomial)


# ============================================================
# 5. SELESAI
# ============================================================

print("\nSemua analisis Progress 2 berhasil dijalankan.")