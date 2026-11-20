
# ============================================================
# PROGRESS 3 - STATISTIKA INFERENSIAL
# KELOMPOK 15 | TEMA B: KUALITAS UDARA
# MATA KULIAH STATISTIKA DAN PROBABILITAS
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
import os

# ============================================================
# 1. MEMUAT DATASET
# ============================================================

os.makedirs("hasil", exist_ok=True)

print("=" * 65)
print("PROGRESS 3 - STATISTIKA INFERENSIAL")
print("KELOMPOK 15 - KUALITAS UDARA")
print("=" * 65)

try:
    df = pd.read_csv("hasil/dataset_bersih.csv")
except FileNotFoundError:
    print("ERROR: File dataset_bersih.csv tidak ditemukan!")
    print("Pastikan file berada di folder hasil.")
    raise SystemExit

kolom_wajib = [
    "Indeks_PM25",
    "Jenis_Hari",
    "Zona_Lokasi",
    "Status_Udara"
]

kolom_hilang = [
    kolom for kolom in kolom_wajib
    if kolom not in df.columns
]

if kolom_hilang:
    print("ERROR: Kolom berikut tidak ditemukan:", kolom_hilang)
    raise SystemExit

# Mengubah PM2.5 menjadi numerik
df["Indeks_PM25"] = pd.to_numeric(
    df["Indeks_PM25"], errors="coerce"
)

df = df.dropna(subset=kolom_wajib)

pm25 = df["Indeks_PM25"]

print("\nJUMLAH DATA:", len(df))
print("\nKATEGORI JENIS HARI:")
print(df["Jenis_Hari"].value_counts())

print("\nKATEGORI ZONA:")
print(df["Zona_Lokasi"].value_counts())

print("\nKATEGORI STATUS UDARA:")
print(df["Status_Udara"].value_counts())


# ============================================================
# 2. INTERVAL KEPERCAYAAN 95% RATA-RATA PM2.5
# ============================================================

print("\n" + "=" * 65)
print("1. INTERVAL KEPERCAYAAN 95% PM2.5")
print("=" * 65)

n = pm25.count()
mean = pm25.mean()
std = pm25.std(ddof=1)

if n > 1 and std > 0:

    standard_error = std / np.sqrt(n)

    t_critical = stats.t.ppf(
        0.975, df=n - 1
    )

    margin_error = t_critical * standard_error

    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    print("Jumlah sampel:", n)
    print("Rata-rata:", mean)
    print("Standar deviasi:", std)
    print("Standard error:", standard_error)
    print("Batas bawah 95%:", lower_bound)
    print("Batas atas 95%:", upper_bound)

    hasil_ci = pd.DataFrame({
        "Jumlah_Sampel": [n],
        "Mean": [mean],
        "Standar_Deviasi": [std],
        "Batas_Bawah_95": [lower_bound],
        "Batas_Atas_95": [upper_bound]
    })

    hasil_ci.to_csv(
        "hasil/interval_kepercayaan_pm25.csv",
        index=False
    )

else:
    print("Data tidak cukup untuk menghitung interval kepercayaan.")


# ============================================================
# 3. UJI T INDEPENDEN WELCH
# PERBANDINGAN HARI KERJA DAN AKHIR PEKAN
# ============================================================

print("\n" + "=" * 65)
print("2. UJI T INDEPENDEN WELCH")
print("=" * 65)

hari_kerja = df[
    df["Jenis_Hari"].astype(str).str.strip().str.lower()
    == "hari kerja"
]["Indeks_PM25"].dropna()

akhir_pekan = df[
    df["Jenis_Hari"].astype(str).str.strip().str.lower()
    == "akhir pekan"
]["Indeks_PM25"].dropna()

if len(hari_kerja) > 1 and len(akhir_pekan) > 1:

    t_stat, p_value_t = stats.ttest_ind(
        hari_kerja,
        akhir_pekan,
        equal_var=False
    )

    print("Jumlah data hari kerja:", len(hari_kerja))
    print("Jumlah data akhir pekan:", len(akhir_pekan))
    print("Mean hari kerja:", hari_kerja.mean())
    print("Mean akhir pekan:", akhir_pekan.mean())
    print("T-statistik:", t_stat)
    print("P-value:", p_value_t)

    if p_value_t < 0.05:
        keputusan_t = "Tolak H0"
        interpretasi_t = (
            "Terdapat perbedaan rata-rata PM2.5 "
            "yang signifikan antara hari kerja "
            "dan akhir pekan."
        )
    else:
        keputusan_t = "Gagal menolak H0"
        interpretasi_t = (
            "Belum terdapat bukti yang cukup "
            "untuk menyatakan adanya perbedaan "
            "rata-rata PM2.5 antara kedua kelompok."
        )

    print("Keputusan:", keputusan_t)
    print("Interpretasi:", interpretasi_t)

    hasil_t = pd.DataFrame({
        "Jumlah_Hari_Kerja": [len(hari_kerja)],
        "Jumlah_Akhir_Pekan": [len(akhir_pekan)],
        "Mean_Hari_Kerja": [hari_kerja.mean()],
        "Mean_Akhir_Pekan": [akhir_pekan.mean()],
        "T_Statistik": [t_stat],
        "P_Value": [p_value_t],
        "Keputusan": [keputusan_t],
        "Interpretasi": [interpretasi_t]
    })

    hasil_t.to_csv(
        "hasil/uji_t_hari_kerja_vs_akhir_pekan.csv",
        index=False
    )

else:
    print("PERHATIAN: Kelompok hari kerja atau akhir pekan")
    print("tidak ditemukan atau jumlah datanya kurang dari 2.")
    print("Periksa nama kategori pada dataset.")


# ============================================================
# 4. ANOVA SATU ARAH ANTAR ZONA
# ============================================================

print("\n" + "=" * 65)
print("3. ANOVA ANTAR ZONA LOKASI")
print("=" * 65)

kelompok_zona = {
    nama: grup["Indeks_PM25"].dropna()
    for nama, grup in df.groupby("Zona_Lokasi")
}

kelompok_valid = [
    data for data in kelompok_zona.values()
    if len(data) > 1
]

if len(kelompok_valid) >= 2:

    f_stat, p_value_anova = stats.f_oneway(
        *kelompok_valid
    )

    print("Jumlah zona:", len(kelompok_valid))
    print("Nilai F:", f_stat)
    print("P-value:", p_value_anova)

    if p_value_anova < 0.05:
        keputusan_anova = "Tolak H0"
        interpretasi_anova = (
            "Terdapat perbedaan rata-rata PM2.5 "
            "yang signifikan pada setidaknya dua zona."
        )
    else:
        keputusan_anova = "Gagal menolak H0"
        interpretasi_anova = (
            "Belum terdapat bukti yang cukup "
            "untuk menyatakan adanya perbedaan "
            "rata-rata PM2.5 antar zona."
        )

    print("Keputusan:", keputusan_anova)
    print("Interpretasi:", interpretasi_anova)

    rata_zona = df.groupby(
        "Zona_Lokasi"
    )["Indeks_PM25"].agg(
        Jumlah_Data="count",
        Rata_Rata="mean",
        Median="median",
        Standar_Deviasi="std"
    ).reset_index()

    rata_zona["F_Statistik"] = f_stat
    rata_zona["P_Value"] = p_value_anova
    rata_zona["Keputusan"] = keputusan_anova

    rata_zona.to_csv(
        "hasil/anova_pm25_antar_zona.csv",
        index=False
    )

else:
    print("Data zona tidak mencukupi untuk ANOVA.")


# ============================================================
# 5. UJI CHI-SQUARE
# HUBUNGAN JENIS HARI DAN STATUS UDARA
# ============================================================

print("\n" + "=" * 65)
print("4. UJI CHI-SQUARE")
print("=" * 65)

tabel_kontingensi = pd.crosstab(
    df["Jenis_Hari"],
    df["Status_Udara"]
)

print("\nTABEL KONTINGENSI:")
print(tabel_kontingensi)

if (
    tabel_kontingensi.shape[0] >= 2
    and tabel_kontingensi.shape[1] >= 2
):

    chi2, p_value_chi, dof, expected = (
        stats.chi2_contingency(tabel_kontingensi)
    )

    print("\nChi-Square:", chi2)
    print("P-value:", p_value_chi)
    print("Degree of Freedom:", dof)

    print(
        "Frekuensi harapan minimum:",
        expected.min()
    )

    if p_value_chi < 0.05:
        keputusan_chi = "Tolak H0"
        interpretasi_chi = (
            "Terdapat hubungan yang signifikan "
            "antara jenis hari dan status udara."
        )
    else:
        keputusan_chi = "Gagal menolak H0"
        interpretasi_chi = (
            "Belum terdapat bukti yang cukup "
            "untuk menyatakan adanya hubungan "
            "antara jenis hari dan status udara."
        )

    print("Keputusan:", keputusan_chi)
    print("Interpretasi:", interpretasi_chi)

    hasil_chi = pd.DataFrame({
        "Chi_Square": [chi2],
        "P_Value": [p_value_chi],
        "Degree_of_Freedom": [dof],
        "Expected_Minimum": [expected.min()],
        "Keputusan": [keputusan_chi],
        "Interpretasi": [interpretasi_chi]
    })

    hasil_chi.to_csv(
        "hasil/uji_chi_square_jenis_hari_status_udara.csv",
        index=False
    )

    tabel_kontingensi.to_csv(
        "hasil/tabel_kontingensi_status_udara.csv"
    )

else:
    print(
        "Data kategori tidak mencukupi "
        "untuk pengujian Chi-Square."
    )


# ============================================================
# 6. RINGKASAN HASIL
# ============================================================

print("\n" + "=" * 65)
print("RINGKASAN OUTPUT PROGRESS 3")
print("=" * 65)

print("1. interval_kepercayaan_pm25.csv")
print("2. uji_t_hari_kerja_vs_akhir_pekan.csv")
print("3. anova_pm25_antar_zona.csv")
print("4. uji_chi_square_jenis_hari_status_udara.csv")
print("5. tabel_kontingensi_status_udara.csv")

print("\nPROGRESS 3 SELESAI!")
print("Periksa folder hasil untuk melihat seluruh output.")
print("=" * 65)