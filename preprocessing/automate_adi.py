import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

def run_preprocessing():
    print("Memulai proses preprocessing otomatis...")
    
    # 1. Load Data Mentah
    # Sesuaikan path karena script ini akan dijalankan dari root repository
    df = pd.read_csv('dataset_raw/synthetic_fooddelivery_dataset.csv')
    df_clean = df.copy()

    # 2. Inisialisasi Kolom
    target_col = 'Status_Pesanan'
    kolom_numerikal = ['Harga_Pesanan', 'Jarak_Kirim_KM', 'Waktu_Tunggu_Menit', 'Rating_Pelanggan']
    kolom_kategorikal = ['Kategori_Menu', 'Status_Promo', 'Tingkat_Keluhan']
    kolom_buang = ['ID_Pesanan', 'Waktu_Transaksi', 'Ulasan_Teks']

    # 3. Menghapus Kolom Tidak Relevan
    df_clean = df_clean.drop(columns=[col for col in kolom_buang if col in df_clean.columns])

    # 4. Imputasi Missing Values
    # Untuk numerikal tetap pakai SimpleImputer
    imputer_num = SimpleImputer(strategy='median')
    df_clean[kolom_numerikal] = imputer_num.fit_transform(df_clean[kolom_numerikal])

    # Untuk kategorikal, gunakan Pandas fillna & mode agar tidak ada error konversi string
    for col in kolom_kategorikal:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 5. Encoding
    df_clean = pd.get_dummies(df_clean, columns=kolom_kategorikal, drop_first=True)
    le = LabelEncoder()
    df_clean[target_col] = le.fit_transform(df_clean[target_col])

    # 6. Standarisasi
    scaler = StandardScaler()
    df_clean[kolom_numerikal] = scaler.fit_transform(df_clean[kolom_numerikal])

    # 7. Simpan Dataset Bersih
    # Membuat folder tujuan jika belum ada
    output_dir = 'preprocessing/dataset_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'food_delivery_cleaned.csv')
    df_clean.to_csv(output_path, index=False)
    
    print(f"Preprocessing selesai! Data bersih disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()