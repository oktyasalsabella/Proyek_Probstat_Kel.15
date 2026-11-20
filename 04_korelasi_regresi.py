# ==========================================
# PROGRESS 4: KORELASI DAN REGRESI
# KELOMPOK 15 - TEMA B: KUALITAS UDARA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. MEMBACA DATA
data = pd.read_csv("hasil/dataset_bersih.csv")

# Variabel yang digunakan
variabel = [
    "Volume_Kendaraan",
    "Kecepatan_Angin_KmJ",
    "Suhu_Celcius",
    "Indeks_PM25"
]

df = data[variabel].dropna()

X = df[[
    "Volume_Kendaraan",
    "Kecepatan_Angin_KmJ",
    "Suhu_Celcius"
]]

Y = df["Indeks_PM25"]

print("DATA BERHASIL DIBACA")
print("Jumlah data:", len(df))


# 2. MATRIKS KORELASI PEARSON
matriks = df.corr(method="pearson")

matriks.to_csv("hasil/matriks_korelasi.csv")

korelasi_pm25 = matriks["Indeks_PM25"].drop(
    "Indeks_PM25"
).reset_index()

korelasi_pm25.columns = ["Variabel", "Korelasi_Pearson"]

korelasi_pm25.to_csv(
    "hasil/korelasi_terhadap_pm25.csv",
    index=False
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    matriks,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matriks Korelasi Pearson")
plt.tight_layout()

plt.savefig("hasil/heatmap_korelasi.png")
plt.close()


# 3. SCATTERPLOT DAN GARIS TREN
daftar_x = [
    "Volume_Kendaraan",
    "Kecepatan_Angin_KmJ",
    "Suhu_Celcius"
]

for x in daftar_x:

    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=df,
        x=x,
        y="Indeks_PM25"
    )

    plt.title(f"Hubungan {x} dengan Indeks PM2.5")
    plt.xlabel(x)
    plt.ylabel("Indeks PM2.5")

    plt.tight_layout()

    plt.savefig(f"hasil/scatterplot_{x}_pm25.png")
    plt.close()


# 4. REGRESI LINEAR BERGANDA
X_const = sm.add_constant(X)

model = sm.OLS(Y, X_const).fit()

# Menyimpan ringkasan model
with open("hasil/ringkasan_regresi.txt", "w") as file:
    file.write(model.summary().as_text())


# 5. PERSAMAAN REGRESI
koefisien = model.params

persamaan = (
    f"Indeks_PM25 = {koefisien['const']:.4f}"
    f" + ({koefisien['Volume_Kendaraan']:.4f}"
    f" * Volume_Kendaraan)"
    f" + ({koefisien['Kecepatan_Angin_KmJ']:.4f}"
    f" * Kecepatan_Angin_KmJ)"
    f" + ({koefisien['Suhu_Celcius']:.4f}"
    f" * Suhu_Celcius)"
)

with open("hasil/persamaan_regresi.txt", "w") as file:
    file.write(persamaan)

print("\nPERSAMAAN REGRESI:")
print(persamaan)


# 6. HASIL PREDIKSI
hasil_prediksi = pd.DataFrame({
    "PM25_Aktual": Y,
    "PM25_Prediksi": model.predict(X_const)
})

hasil_prediksi.to_csv(
    "hasil/hasil_regresi.csv",
    index=False
)


# 7. R2 DAN ADJUSTED R2
evaluasi = pd.DataFrame({
    "R_squared": [model.rsquared],
    "Adjusted_R_squared": [model.rsquared_adj],
    "F_statistic": [model.fvalue],
    "P_value_F": [model.f_pvalue],
    "Jumlah_Data": [len(df)]
})

evaluasi.to_csv(
    "hasil/evaluasi_model_regresi.csv",
    index=False
)


# 8. UJI F SIMULTAN
uji_f = pd.DataFrame({
    "F_statistic": [model.fvalue],
    "P_value": [model.f_pvalue],
    "Keputusan": [
        "Signifikan" if model.f_pvalue < 0.05
        else "Tidak Signifikan"
    ]
})

uji_f.to_csv(
    "hasil/uji_f_simultan.csv",
    index=False
)


# 9. UJI T PARSIAL
uji_t = pd.DataFrame({
    "Koefisien": model.params,
    "T_statistic": model.tvalues,
    "P_value": model.pvalues
})

uji_t["Keputusan"] = np.where(
    uji_t["P_value"] < 0.05,
    "Signifikan",
    "Tidak Signifikan"
)

uji_t.to_csv(
    "hasil/uji_t_parsial.csv"
)


# 10. ANALISIS RESIDUAL
residual = model.resid
prediksi = model.predict(X_const)

# Histogram residual
plt.figure(figsize=(8, 5))

plt.hist(residual, bins=20, edgecolor="black")

plt.title("Histogram Residual")
plt.xlabel("Residual")
plt.ylabel("Frekuensi")

plt.tight_layout()

plt.savefig("hasil/histogram_residual.png")
plt.close()


# Q-Q Plot
plt.figure(figsize=(8, 5))

sm.qqplot(
    residual,
    line="45",
    fit=True
)

plt.title("Q-Q Plot Residual")

plt.tight_layout()

plt.savefig("hasil/qqplot_residual.png")
plt.close()


# Fitted vs Residual
plt.figure(figsize=(8, 5))

plt.scatter(prediksi, residual)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Fitted Values vs Residual")
plt.xlabel("Nilai Prediksi")
plt.ylabel("Residual")

plt.tight_layout()

plt.savefig("hasil/fitted_vs_residuals.png")
plt.close()


# 11. UJI NORMALITAS RESIDUAL
shapiro = stats.shapiro(residual)

# 12. UJI HOMOSKEDASTISITAS
bp = het_breuschpagan(
    residual,
    X_const
)

diagnostik = pd.DataFrame({
    "Pengujian": [
        "Shapiro-Wilk",
        "Breusch-Pagan"
    ],
    "Statistik": [
        shapiro.statistic,
        bp[0]
    ],
    "P_value": [
        shapiro.pvalue,
        bp[1]
    ],
    "Interpretasi": [
        "Normal" if shapiro.pvalue >= 0.05
        else "Indikasi tidak normal",
        "Homoskedastisitas" if bp[1] >= 0.05
        else "Indikasi heteroskedastisitas"
    ]
})

diagnostik.to_csv(
    "hasil/diagnostik_residual.csv",
    index=False
)


# 13. REKOMENDASI AWAL
rekomendasi = []

rekomendasi.append(
    "Analisis hubungan dan model regresi tidak membuktikan "
    "hubungan sebab-akibat. Hasil perlu ditafsirkan sesuai "
    "keterbatasan data."
)

rekomendasi.append(
    "Variabel yang memiliki hubungan statistik dengan PM2.5 "
    "dapat menjadi bahan pertimbangan untuk pemantauan "
    "kualitas udara lebih lanjut."
)

rekomendasi.append(
    "Pengembangan model selanjutnya dapat mempertimbangkan "
    "variabel tambahan, pengumpulan data yang lebih luas, "
    "serta evaluasi model pada data berbeda."
)

with open("hasil/rekomendasi_awal.txt", "w") as file:
    for i, item in enumerate(rekomendasi, start=1):
        file.write(f"{i}. {item}\n")


# 14. OUTPUT TERMINAL
print("\n====================================")
print("PROGRESS 4 BERHASIL DIJALANKAN")
print("====================================")

print("\nR-squared:", model.rsquared)
print("Adjusted R-squared:", model.rsquared_adj)
print("F-statistic:", model.fvalue)
print("P-value F:", model.f_pvalue)

print("\nUJI NORMALITAS RESIDUAL")
print("Shapiro Statistic:", shapiro.statistic)
print("Shapiro P-value:", shapiro.pvalue)

print("\nUJI HOMOSKEDASTISITAS")
print("Breusch-Pagan Statistic:", bp[0])
print("Breusch-Pagan P-value:", bp[1])

print("\nSemua hasil disimpan di folder hasil.")