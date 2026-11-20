# ============================================================
# PROGRESS 3 - INFERENSI STATISTIK
# Kelompok 15 - Tema B: Kualitas Udara
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# 1. MEMBACA DATASET
# ============================================================

data = pd.read_csv("hasil/dataset_bersih.csv")

print("Jumlah data:", len(data))


# ============================================================
# 2. CONFIDENCE INTERVAL 95% RATA-RATA PM2.5
# ============================================================

pm25 = data["Indeks_PM25"].dropna()

n = len(pm25)
mean_pm25 = pm25.mean()
std_pm25 = pm25.std(ddof=1)

alpha = 0.05

# Menggunakan distribusi t
t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

margin_error = t_critical * (std_pm25 / np.sqrt(n))

lower_mean = mean_pm25 - margin_error
upper_mean = mean_pm25 + margin_error

hasil_ci_mean = pd.DataFrame([{
    "Variabel": "Indeks_PM25",
    "Jumlah Data": n,
    "Mean": mean_pm25,
    "Standar Deviasi": std_pm25,
    "Confidence Level": "95%",
    "Batas Bawah": lower_mean,
    "Batas Atas": upper_mean
}])

hasil_ci_mean.to_csv(
    "hasil/progress3_ci_mean_pm25.csv",
    index=False
)

print("\n=== CONFIDENCE INTERVAL RATA-RATA PM2.5 ===")
print(hasil_ci_mean)


# ============================================================
# 3. CONFIDENCE INTERVAL 95% PROPORSI STATUS TIDAK SEHAT
# ============================================================

jumlah_data = len(data)

jumlah_tidak_sehat = (
    data["Status_Udara"] == "Tidak Sehat"
).sum()

p_hat = jumlah_tidak_sehat / jumlah_data

# Metode Wilson untuk CI proporsi
z = stats.norm.ppf(1 - alpha / 2)

denominator = 1 + (z**2 / jumlah_data)

center = (
    p_hat + (z**2 / (2 * jumlah_data))
) / denominator

half_width = (
    z * np.sqrt(
        (p_hat * (1 - p_hat) / jumlah_data)
        + (z**2 / (4 * jumlah_data**2))
    )
) / denominator

lower_prop = center - half_width
upper_prop = center + half_width

hasil_ci_proporsi = pd.DataFrame([{
    "Kategori": "Tidak Sehat",
    "Jumlah": jumlah_tidak_sehat,
    "Total Data": jumlah_data,
    "Proporsi": p_hat,
    "Confidence Level": "95%",
    "Batas Bawah": lower_prop,
    "Batas Atas": upper_prop
}])

hasil_ci_proporsi.to_csv(
    "hasil/progress3_ci_proporsi.csv",
    index=False
)

print("\n=== CONFIDENCE INTERVAL PROPORSI ===")
print(hasil_ci_proporsi)


# ============================================================
# 4. INDEPENDENT TWO-SAMPLE T-TEST
# Hari Kerja vs Akhir Pekan
# ============================================================

hari_kerja = data[
    data["Jenis_Hari"] == "Hari Kerja"
]["Indeks_PM25"].dropna()

akhir_pekan = data[
    data["Jenis_Hari"] == "Akhir Pekan"
]["Indeks_PM25"].dropna()

# Welch's t-test (tidak mengasumsikan varians sama)
t_stat, p_value_t = stats.ttest_ind(
    hari_kerja,
    akhir_pekan,
    equal_var=False
)

hasil_t_test = pd.DataFrame([{
    "Kelompok 1": "Hari Kerja",
    "Kelompok 2": "Akhir Pekan",
    "Mean Hari Kerja": hari_kerja.mean(),
    "Mean Akhir Pekan": akhir_pekan.mean(),
    "Statistik t": t_stat,
    "P_Value": p_value_t,
    "Alpha": alpha,
    "Keputusan": (
        "Tolak H0"
        if p_value_t < alpha
        else "Gagal menolak H0"
    )
}])

hasil_t_test.to_csv(
    "hasil/progress3_t_test.csv",
    index=False
)

print("\n=== INDEPENDENT T-TEST ===")
print(hasil_t_test)


# ============================================================
# 5. ONE-WAY ANOVA
# PM2.5 berdasarkan Zona_Lokasi
# ============================================================

kampus = data[
    data["Zona_Lokasi"] == "Kawasan Kampus"
]["Indeks_PM25"].dropna()

pusat = data[
    data["Zona_Lokasi"] == "Pusat Kota"
]["Indeks_PM25"].dropna()

pinggiran = data[
    data["Zona_Lokasi"] == "Pinggiran"
]["Indeks_PM25"].dropna()

f_stat, p_value_anova = stats.f_oneway(
    kampus,
    pusat,
    pinggiran
)

hasil_anova = pd.DataFrame([{
    "Variabel": "Indeks_PM25",
    "Kelompok": "Zona_Lokasi",
    "Mean Kampus": kampus.mean(),
    "Mean Pusat Kota": pusat.mean(),
    "Mean Pinggiran": pinggiran.mean(),
    "Statistik F": f_stat,
    "P_Value": p_value_anova,
    "Alpha": alpha,
    "Keputusan": (
        "Tolak H0"
        if p_value_anova < alpha
        else "Gagal menolak H0"
    )
}])

hasil_anova.to_csv(
    "hasil/progress3_anova.csv",
    index=False
)

print("\n=== ONE-WAY ANOVA ===")
print(hasil_anova)


# ============================================================
# 6. POST-HOC TUKEY (JIKA ANOVA SIGNIFIKAN)
# ============================================================

if p_value_anova < alpha:

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey = pairwise_tukeyhsd(
        endog=data["Indeks_PM25"],
        groups=data["Zona_Lokasi"],
        alpha=alpha
    )

    hasil_tukey = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    hasil_tukey.to_csv(
        "hasil/progress3_posthoc_tukey.csv",
        index=False
    )

    print("\n=== POST-HOC TUKEY ===")
    print(hasil_tukey)

else:
    print("\nANOVA tidak signifikan.")
    print("Post-Hoc Tukey tidak dilakukan.")


# ============================================================
# 7. CHI-SQUARE TEST OF INDEPENDENCE
# Jenis_Hari vs Status_Udara
# ============================================================

tabel_observasi = pd.crosstab(
    data["Jenis_Hari"],
    data["Status_Udara"]
)

chi2, p_value_chi, df_chi, expected = (
    stats.chi2_contingency(tabel_observasi)
)

tabel_ekspektasi = pd.DataFrame(
    expected,
    index=tabel_observasi.index,
    columns=tabel_observasi.columns
)

tabel_observasi.to_csv(
    "hasil/progress3_chi_observasi.csv"
)

tabel_ekspektasi.to_csv(
    "hasil/progress3_chi_ekspektasi.csv"
)

hasil_chi = pd.DataFrame([{
    "Variabel 1": "Jenis_Hari",
    "Variabel 2": "Status_Udara",
    "Statistik Chi-Square": chi2,
    "Derajat Bebas": df_chi,
    "P_Value": p_value_chi,
    "Alpha": alpha,
    "Keputusan": (
        "Tolak H0"
        if p_value_chi < alpha
        else "Gagal menolak H0"
    )
}])

hasil_chi.to_csv(
    "hasil/progress3_chi_square.csv",
    index=False
)

print("\n=== TABEL OBSERVASI CHI-SQUARE ===")
print(tabel_observasi)

print("\n=== TABEL EKSPEKTASI CHI-SQUARE ===")
print(tabel_ekspektasi)

print("\n=== CHI-SQUARE TEST ===")
print(hasil_chi)


# ============================================================
# 8. SELESAI
# ============================================================

print("\nSemua analisis Progress 3 berhasil dijalankan.")