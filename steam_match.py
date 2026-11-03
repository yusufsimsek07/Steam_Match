#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
SteamMatch: Yorumlara Dayalı Oyun Öneri Sistemi (Game Recommendation System)
========================================================================================
Ders         : Veri Madenciliği (Data Mining)
Proje Türü   : İşbirlikçi Filtreleme Tabanlı Öneri Sistemi (Collaborative Filtering)
Veri Seti    : Steam Review Dataset (Antoni Sobkowicz) — 21.7+ Milyon Satır
Kütüphane    : scikit-surprise, pandas, numpy, matplotlib
Algoritmalar : 1. SVD (Singular Value Decomposition)
               2. NMF (Non-negative Matrix Factorization)
               3. KNNBasic (Cosine Benzerliği - Item-based)
               4. NormalPredictor (Rastgele Tahmin - Baseline)
Değerlendirme: 3-Fold Cross-Validation (RMSE ve MAE Metrikleri)
========================================================================================
"""

import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surprise import SVD, NMF, KNNBasic, NormalPredictor, Dataset, Reader
from surprise.model_selection import cross_validate

warnings.filterwarnings("ignore")

# ======================================================================================
# 1. BAŞLIK VE BİLGİLENDİRME
# ======================================================================================
print("=" * 85)
print("   🎮 SteamMatch: Yorumlara Dayalı Oyun Öneri Sistemi")
print("   📚 Veri Madenciliği Dersi — Dönem Projesi")
print("=" * 85)

# ======================================================================================
# 2. VERİ YÜKLEME VE BELLEK OPTİMİZASYONU (usecols & dtype)
# ======================================================================================
# Hocanın Kriteri:
# - Metin sütunları (review) belleğe ALINMAYACAK (RAM optimizasyonu).
# - Sadece user_id ('author.steamid'), game_id ('app_id') ve sentiment ('recommended')
#   sütunları okunacaktır.
# ======================================================================================
DATA_PATH = "data/steam_reviews.csv"
if not os.path.exists(DATA_PATH):
    if os.path.exists("data/dataset.csv"):
        DATA_PATH = "data/dataset.csv"
    else:
        raise FileNotFoundError(f"Veri seti 'data/' dizini altında bulunamadı!")

print(f"\n[Adım 1/6] Veri seti yükleniyor: {DATA_PATH}")
print("           RAM optimizasyonu için sadece gerekli sütunlar belleğe alınıyor...")

start_load_time = time.time()

# Sütun isimleri ve düşük bellek tüketen veri tipleri
use_columns = ["author.steamid", "app_id", "recommended"]
dtypes = {
    "author.steamid": "int64",
    "app_id": "int32",
    "recommended": "bool"
}

df = pd.read_csv(
    DATA_PATH,
    usecols=use_columns,
    dtype=dtypes
)

# Sütun isimlerini standartlaştır
df = df.rename(columns={"author.steamid": "user_id", "app_id": "game_id"})

load_duration = time.time() - start_load_time
mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

print(f"   ✓ Toplam Satır Sayısı : {len(df):,}")
print(f"   ✓ Yükleme Süresi       : {load_duration:.2f} saniye")
print(f"   ✓ Bellek (RAM) Kullanımı: {mem_usage_mb:.2f} MB (Metinler elendiği için çok verimli)")

# ======================================================================================
# 3. VERİ ÖN İŞLEME (Data Preprocessing)
# ======================================================================================
# Hocanın Kriterleri:
# a) Sentiment Dönüşümü: Positive (True) -> 5 Puan, Negative (False) -> 1 Puan
# b) Filtreleme:
#    - En az 50 yorum almış oyunlar
#    - En az 5 farklı oyuna yorum yapmış kullanıcılar
# c) Boyut Kriteri: Sonuç veri seti en az 1.000.000 satır olmalıdır!
# ======================================================================================
print("\n[Adım 2/6] Veri Ön İşleme ve Filtreleme Adımları Uygulanıyor...")

# 3a. Sentiment -> 1-5 Rating Matrisi Dönüşümü
print("   • Yorum duygu durumu (sentiment) derecelendirmeye dönüştürülüyor...")
df["rating"] = df["recommended"].map({True: 5, False: 1}).astype("int8")
df = df.drop(columns=["recommended"])

# Tekrarlanan aynı kullanıcı - aynı oyun değerlendirmelerini tekilleştir
before_dup = len(df)
df = df.drop_duplicates(subset=["user_id", "game_id"], keep="last")
print(f"   • Tekilleştirme sonrası satır sayısı: {len(df):,} (Temizlenen tekrar: {before_dup - len(df):,})")

# 3b. Filtreleme 1: En az 50 yorum almış oyunlar
print("   • En az 50 yorum alan popüler oyunlar filtreleniyor...")
game_counts = df["game_id"].value_counts()
valid_games = game_counts[game_counts >= 50].index
df = df[df["game_id"].isin(valid_games)]
print(f"     -> Kalan oyun sayısı: {df['game_id'].nunique():,}, Satır: {len(df):,}")

# 3c. Filtreleme 2: En az 5 farklı oyuna yorum yapmış aktif kullanıcılar
print("   • En az 5 farklı oyuna yorum yapan aktif kullanıcılar filtreleniyor...")
user_counts = df.groupby("user_id")["game_id"].count()
valid_users = user_counts[user_counts >= 5].index
df = df[df["user_id"].isin(valid_users)]
print(f"     -> Kalan kullanıcı sayısı: {df['user_id'].nunique():,}, Satır: {len(df):,}")

# 3d. 1 Milyon Satır Kriteri Kontrolü (Hocanın Şartı)
assert len(df) >= 1_000_000, f"HATA: Filtrelenmiş satır sayısı {len(df):,} < 1.000.000! Eşikler gevşetilmeli."
print(f"\n   ✅ ZORUNLU KRİTER SAĞLANDI: Filtrelenmiş veri {len(df):,} satır (>= 1.000.000 şartı sağlandı).")

# Model eğitimi için tam 1.000.000 satırlık deterministik örneklem seçelim
# (Hem hocanın 1M kriteri birebir karşılanır hem de 3-fold cross-validation optimum sürede biter)
TARGET_ROWS = 1_000_000
df_model = df.sample(n=TARGET_ROWS, random_state=42).reset_index(drop=True)
print(f"   • Modelleme için seçilen örneklem: {len(df_model):,} satır")
print(f"     - Benzersiz Kullanıcı Sayısı : {df_model['user_id'].nunique():,}")
print(f"     - Benzersiz Oyun Sayısı       : {df_model['game_id'].nunique():,}")
print(f"     - Rating Dağılımı             : 5 Puan: {(df_model['rating']==5).sum():,} | 1 Puan: {(df_model['rating']==1).sum():,}")

# Bellek temizliği
del df
import gc
gc.collect()

# ======================================================================================
# 4. SURPRISE FORMATINA DÖNÜŞTÜRME
# ======================================================================================
print("\n[Adım 3/6] scikit-surprise Veri Yapısı Oluşturuluyor...")
# 1-5 rating ölçeği tanımlanıyor
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(df_model[["user_id", "game_id", "rating"]], reader)
print("   ✓ Surprise veri kümesi başarıyla oluşturuldu.")

# ======================================================================================
# 5. ALGORİTMALARIN TANIMLANMASI VE 3-FOLD CROSS-VALIDATION
# ======================================================================================
# Karşılaştırılacak 4 Algoritma:
# 1. SVD (Singular Value Decomposition) - Matris Ayrıştırma
# 2. NMF (Non-negative Matrix Factorization) - Negatif Olmayan Matris Ayrıştırma
# 3. KNNBasic (Cosine Benzerliği, Item-based) - Komşuluk Tabanlı İşbirlikçi Filtreleme
# 4. NormalPredictor (Baseline) - Rastgele Normal Dağılımla Tahmin (Referans Modeli)
# ======================================================================================
print("\n[Adım 4/6] 4 Farklı Algoritma Tanımlanıyor ve 3-Fold Cross-Validation Başlatılıyor...")

algorithms = {
    "NormalPredictor (Baseline)": {
        "model": NormalPredictor(),
        "aciklama": "Normal dağılım tabanlı rastgele tahmin yapan referans taban modeli."
    },
    "KNNBasic (Cosine)": {
        "model": KNNBasic(
            k=20,
            sim_options={"name": "cosine", "user_based": False},  # Item-based Cosine
            verbose=False
        ),
        "aciklama": "Oyunlar arası Kosinüs benzerliğine dayalı komşuluk tabanlı işbirlikçi filtreleme."
    },
    "NMF": {
        "model": NMF(n_factors=15, n_epochs=15, random_state=42),
        "aciklama": "Negatif olmayan gizli özellik matrisleri (W ve H) öğrenen ayrıştırma algoritması."
    },
    "SVD": {
        "model": SVD(n_factors=20, n_epochs=15, random_state=42),
        "aciklama": "Kullanıcı ve oyun gizli faktörlerini öğrenen düzenlileştirilmiş tekil değer ayrıştırması."
    }
}

cv_results_summary = []

for name, meta in algorithms.items():
    print(f"\n{'='*75}")
    print(f"  ▶ Model: {name}")
    print(f"    Tanım: {meta['aciklama']}")
    print(f"{'='*75}")
    
    t_start = time.time()
    
    # 3-Fold Cross-Validation (cv=3)
    cv_output = cross_validate(
        meta["model"],
        surprise_data,
        measures=["RMSE", "MAE"],
        cv=3,
        verbose=False,
        n_jobs=-1  # Paralel işlemci çekirdekleri
    )
    
    elapsed = time.time() - t_start
    
    mean_rmse = np.mean(cv_output["test_rmse"])
    std_rmse = np.std(cv_output["test_rmse"])
    mean_mae = np.mean(cv_output["test_mae"])
    std_mae = np.std(cv_output["test_mae"])
    
    print(f"    ✓ 3-Fold CV Tamamlandı: {elapsed:.2f} saniye")
    print(f"    • Test RMSE : {mean_rmse:.4f} (±{std_rmse:.4f})")
    print(f"    • Test MAE  : {mean_mae:.4f} (±{std_mae:.4f})")
    
    cv_results_summary.append({
        "Algoritma": name,
        "Test RMSE": round(mean_rmse, 4),
        "RMSE Std": round(std_rmse, 4),
        "Test MAE": round(mean_mae, 4),
        "MAE Std": round(std_mae, 4),
        "Eğitim Süresi (sn)": round(elapsed, 2)
    })

# ======================================================================================
# 6. SONUÇLARIN DATAFRAME OLARAK RAPORLANMASI
# ======================================================================================
print("\n" + "=" * 85)
print("  🏆 KARŞILAŞTIRMALI MODEL PERFORMANS TABLOSU (3-Fold Cross-Validation)")
print("=" * 85)

results_df = pd.DataFrame(cv_results_summary)
# RMSE'ye göre küçükten büyüğe sırala (en başarılı en üstte)
results_df = results_df.sort_values(by="Test RMSE", ascending=True).reset_index(drop=True)

print(results_df.to_string(index=False))

# Çıktıyı outputs klasörüne kaydet
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/sonuclar.csv", index=False, encoding="utf-8-sig")
print("\n   ✓ Sonuçlar 'outputs/sonuclar.csv' dosyasına kaydedildi.")

# ======================================================================================
# 7. GÖRSELLEŞTİRME — PROFESYONEL SÜTUN GRAFİĞİ (Bar Chart)
# ======================================================================================
print("\n[Adım 5/6] Karşılaştırmalı Performans Grafikleri Çiziliyor...")

# Şık renk paleti
colors = ["#27ae60", "#2980b9", "#f39c12", "#c0392b"]

# Grafik 1: RMSE Karşılaştırması
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

algo_names = [name.replace(" ", "\n") for name in results_df["Algoritma"]]
rmse_vals = results_df["Test RMSE"].tolist()
rmse_errs = results_df["RMSE Std"].tolist()

bars = ax.bar(
    algo_names,
    rmse_vals,
    yerr=rmse_errs,
    capsize=7,
    color=colors[:len(algo_names)],
    edgecolor="#2c3e50",
    linewidth=1.2,
    width=0.55,
    alpha=0.9
)

# Sütun tepe değerlerini yazdır
for bar, val in zip(bars, rmse_vals):
    height = bar.get_height()
    ax.annotate(
        f"{val:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#2c3e50"
    )

ax.set_title("SteamMatch: Algoritmaların RMSE Karşılaştırması\n(3-Fold CV | 1.000.000 Satır Steam Verisi)", 
             fontsize=14, fontweight="bold", pad=15, color="#2c3e50")
ax.set_xlabel("Öneri Algoritması", fontsize=12, fontweight="semibold", labelpad=10)
ax.set_ylabel("RMSE (Kök Ortalama Kare Hata — Düşük Olması İyidir)", fontsize=11, labelpad=10)
ax.set_ylim(0, max(rmse_vals) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# En iyi modeli belirten vurgu
best_algo = results_df.iloc[0]["Algoritma"]
best_rmse = results_df.iloc[0]["Test RMSE"]
baseline_rmse = results_df[results_df["Algoritma"].str.contains("Baseline")]["Test RMSE"].values[0]
improvement_pct = ((baseline_rmse - best_rmse) / baseline_rmse) * 100

ax.text(
    0.03, 0.92,
    f"★ En İyi Model: {best_algo} (RMSE: {best_rmse:.4f})\n"
    f"★ Baseline Modeline Göre Hata Azalması: %{improvement_pct:.1f}",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7", alpha=0.9)
)

plt.tight_layout()
rmse_fig_path = "outputs/rmse_karsilastirma.png"
plt.savefig(rmse_fig_path, dpi=300)
print(f"   ✓ RMSE Grafiği kaydedildi: {rmse_fig_path}")
plt.close()

# Grafik 2: İkili Karşılaştırma (RMSE ve MAE yan yana)
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# RMSE subplot
bars1 = ax1.bar(algo_names, rmse_vals, color=colors, edgecolor="#2c3e50", width=0.55)
for bar, val in zip(bars1, rmse_vals):
    ax1.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold", fontsize=11)
ax1.set_title("RMSE Karşılaştırması", fontsize=13, fontweight="bold")
ax1.set_ylabel("RMSE", fontsize=11)
ax1.set_ylim(0, max(rmse_vals) * 1.22)
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# MAE subplot
mae_vals = results_df["Test MAE"].tolist()
bars2 = ax2.bar(algo_names, mae_vals, color=colors, edgecolor="#2c3e50", width=0.55)
for bar, val in zip(bars2, mae_vals):
    ax2.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold", fontsize=11)
ax2.set_title("MAE (Ortalama Mutlak Hata) Karşılaştırması", fontsize=13, fontweight="bold")
ax2.set_ylabel("MAE", fontsize=11)
ax2.set_ylim(0, max(mae_vals) * 1.22)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

fig2.suptitle("SteamMatch: Model Doğruluk Metrikleri (RMSE & MAE)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
combo_fig_path = "outputs/model_karsilastirma.png"
plt.savefig(combo_fig_path, dpi=300)
print(f"   ✓ RMSE & MAE Birleşik Grafiği kaydedildi: {combo_fig_path}")
plt.close()

# ======================================================================================
# 8. ÖRNEK OYUN ÖNERİSİ ÇIKTISI (Sunum İçin İlave Değer)
# ======================================================================================
print("\n[Adım 6/6] En İyi Model (SVD) Eğitiliyor ve Örnek Kullanıcıya Oyun Önerisi Üretiliyor...")

# En başarılı model olan SVD'yi tüm 1M veri üzerinde fit edelim
full_trainset = surprise_data.build_full_trainset()
svd_model = SVD(n_factors=20, n_epochs=15, random_state=42)
svd_model.fit(full_trainset)

# Örnek bir kullanıcı seçip oynamadığı oyunlar arasından en yüksek tahminli 5 oyunu önerelim
sample_user_id = df_model["user_id"].iloc[0]
user_rated_games = set(df_model[df_model["user_id"] == sample_user_id]["game_id"])
all_unique_games = df_model["game_id"].unique()
unseen_games = [g for g in all_unique_games if g not in user_rated_games]

predictions = [svd_model.predict(sample_user_id, g) for g in unseen_games]
predictions.sort(key=lambda x: x.est, reverse=True)

print(f"\n   👤 Örnek Kullanıcı ID: {sample_user_id}")
print(f"   🎯 Kullanıcı için En Yüksek Puan Tahmini Yapılan Top 5 Oyun:")
for rank, pred in enumerate(predictions[:5], start=1):
    print(f"      {rank}. Oyun (App ID): {pred.iid:<10} -> Tahmini Puan: {pred.est:.2f} / 5.00")

print("\n" + "=" * 85)
print("  ✅ PROJE ÇALIŞMASI BAŞARIYLA TAMAMLANDI!")
print("  Tüm grafikler ve sonuç tabloları 'outputs/' klasörüne kaydedilmiştir.")
print("=" * 85)

