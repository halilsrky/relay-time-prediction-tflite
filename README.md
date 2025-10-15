# Elektrik Koruma Rölesi TIME Tahmin Modelleri

Bu proje, elektrik koruma rölesinin tepki süresini (TIME) tahmin etmek için çeşitli makine öğrenmesi modellerini geliştirmekte ve bunları STM32MP1 gibi embedded sistemlerde kullanılabilir formatlara dönüştürmektedir.

## 📁 Proje Yapısı

```
MODELS/
├── 📄 Python Scripts
│   ├── model_comparison.py          # 4 farklı ML modelini karşılaştırma
│   ├── tflite_model_converter.py    # Neural Network → TensorFlow Lite
│   ├── m2cgen_xgboost_converter.py  # XGBoost → TensorFlow Lite (M2Cgen)
│   └── get_scaler_params.py         # StandardScaler parametrelerini çıkarma
│
├── 🎯 C++ Test Dosyaları
│   ├── neural_network_test.cpp      # Neural Network TFLite test
│   └── xgboost_equivalent_test.cpp  # XGBoost TFLite test
│
├── 📊 Veri Dosyaları
│   ├── train.xlsx                   # Eğitim verisi
│   ├── test.xlsx                    # Test verisi
│   ├── input.csv                    # Test giriş verileri
│   └── targets.csv                  # Test hedef değerleri
│
├── 🗂️ Klasörler
│   ├── saved_models/               # PKL model dosyaları
│   ├── tflite_models/              # TensorFlow Lite modelleri
│   └── model_outputs/              # Grafik ve raporlar
│
└── 📋 Raporlar
    ├── README.md                    # Bu dosya
    └── ELEKTRIK_KORUMA_ROLESI_TEKNIK_RAPOR.md
```

## 🚀 Ana Özellikler

### 1. Model Karşılaştırması (`model_comparison.py`)
- **4 farklı algoritma** karşılaştırması:
  - Linear Regression
  - XGBoost
  - LightGBM
  - Neural Network (MLP)
- **Kapsamlı analiz** ile overfitting tespiti
- **Görselleştirmeler** ve detaylı raporlama

### 2. TensorFlow Lite Dönüşümü (`tflite_model_converter.py`)
- **Neural Network** modelini TFLite formatına dönüştürme
- **Embedded sistem** optimizasyonu
- **StandardScaler** parametreleri kaydetme
- **Test ve doğrulama** işlemleri

### 3. XGBoost Dönüşümü (`m2cgen_xgboost_converter.py`)
- **M2Cgen** kütüphanesi ile XGBoost → TFLite
- **Kod tabanlı** dönüşüm
- **Karşılaştırmalı test** işlemleri

### 4. C++ Test Uygulamaları
- **TensorFlow Lite C API** kullanımı
- **Gerçek zamanlı** tahmin testleri
- **Performans** ölçümleri

## 📈 Model Performansları

### Neural Network (TensorFlow Lite)
- **R² Score:** 0.9796
- **RMSE:** 0.0462
- **MAE:** 0.0379
- **Model Boyutu:** 22.1 KB

### XGBoost (M2Cgen → TFLite)
- Embedded sistem uyumlu format
- Yüksek doğruluk oranı
- Düşük bellek kullanımı

## 🔧 Kurulum ve Kullanım

### Gereksinimler
```bash
pip install pandas numpy scikit-learn tensorflow xgboost lightgbm joblib openpyxl matplotlib seaborn m2cgen
```

### 1. Model Karşılaştırması Çalıştırma
```bash
python3 model_comparison.py
```
**Çıktılar:**
- `saved_models/` - Eğitilmiş modeller (.pkl)
- `model_outputs/` - Grafikler ve analizler

### 2. Neural Network TFLite Dönüşümü
```bash
python3 tflite_model_converter.py
```
**Çıktılar:**
- `tflite_models/tensorflow_nn_model.tflite` - Ana model
- `tflite_models/scaler.pkl` - Normalizasyon parametreleri
- `tflite_models/usage_example.py` - Kullanım örneği

### 3. XGBoost TFLite Dönüşümü
```bash
python3 m2cgen_xgboost_converter.py
```
**Çıktılar:**
- `tflite_models/xgboost_m2cgen.tflite` - XGBoost TFLite modeli
- `tflite_models/xgboost_m2cgen_model.pkl` - PKL versiyonu

### 4. C++ Test Uygulamaları
```bash
# Neural Network testi
g++ -o neural_test neural_network_test.cpp -ltensorflowlite
./neural_test

# XGBoost testi
g++ -o xgboost_test xgboost_equivalent_test.cpp -ltensorflowlite
./xgboost_test
```

## 📊 Çıktı Dosyaları

### Kaydedilen Modeller (`saved_models/`)
- `linear_regression_model.pkl`
- `xgboost_model.pkl`
- `lightgbm_model.pkl`
- `neural_network_model.pkl`
- `scaler.pkl`

### TensorFlow Lite Modelleri (`tflite_models/`)
- `tensorflow_nn_model.tflite` (22.1 KB)
- `xgboost_m2cgen.tflite`
- `scaler.pkl`
- `feature_names.txt`
- `usage_example.py`

### Analiz Raporları (`model_outputs/`)
- `comprehensive_model_report.xlsx` - Kapsamlı Excel raporu
- `model_comparison_results.png` - Performans karşılaştırması
- `feature_importance.png` - Özellik önemleri
- `actual_vs_predicted.png` - Tahmin doğruluğu
- `overfitting_analysis.png` - Aşırı öğrenme analizi
- Ve daha fazlası...

## 🎯 Embedded Sistem Kullanımı

### Feature Format
```cpp
// Input features: [Ip, TMS, IL, Isc, PWT1, PWT3, FL, Ftype_2, Ftype_3, Ftype_4]
float features[10] = {110.58, 0.1, 92.15, 5893.56, 2.5, 2.5, 1.0, 0, 0, 0};
```

### Neural Network (Scaling Gerekli)
```cpp
// 1. StandardScaler ile normalizasyon
// 2. TFLite model yükleme
// 3. Tahmin yapma
```

### XGBoost (Scaling Gerekmez)
```cpp
// 1. Doğrudan TFLite model yükleme
// 2. Raw feature değerleri ile tahmin
```

## 📋 Teknik Detaylar

### Veri Seti
- **Training:** 626,294 örnek
- **Test:** 156,574 örnek
- **Features:** Ip, TMS, IL, Isc, PWT1, PWT3, FL, Ftype
- **Target:** TIME (saniye)

### Feature Engineering
- **One-hot encoding** Ftype kategorik değişkeni için
- **StandardScaler** Neural Network için
- **Raw değerler** XGBoost için

### Model Optimizasyonları
- **Early Stopping** overfitting önlemi
- **TensorFlow Lite** embedded optimize
- **Quantization** boyut optimizasyonu

## 🔍 Kalite Kontrol

### Test Metrikleri
- **R² Score** - Açıklanan varyans oranı
- **RMSE** - Kök ortalama kare hatası
- **MAE** - Ortalama mutlak hata
- **Overfitting Score** - Train vs Test R² farkı

### Doğrulama
- **Cross-validation** model güvenilirliği
- **TFLite test** embedded uyumluluk
- **C++ validation** gerçek zamanlı performans

## 🎓 Kullanım Örnekleri

### Python'da TFLite Kullanımı
```python
import tensorflow as tf
import numpy as np
import joblib

# Model ve scaler yükleme
interpreter = tf.lite.Interpreter(model_path="tflite_models/tensorflow_nn_model.tflite")
scaler = joblib.load("tflite_models/scaler.pkl")

# Tahmin yapma
features = [110.58, 0.1, 92.15, 5893.56, 2.5, 2.5, 1.0, 0, 0, 0]
scaled_features = scaler.transform([features])
# ... TFLite tahmin işlemi
```

### C++'da TFLite Kullanımı
```cpp
#include <tensorflow/lite/c/c_api.h>

// Model yükleme ve tahmin
TfLiteModel* model = TfLiteModelCreateFromFile("tensorflow_nn_model.tflite");
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, nullptr);
// ... tahmin işlemleri
```

## 📞 Destek

Bu proje elektrik koruma röleleri için geliştirilmiş özel bir TIME tahmin sistemidir. STM32MP1 embedded sistemlerde çalışacak şekilde optimize edilmiştir.

**Geliştirici:** Halil
**Proje Tarihi:** Ekim 2025
**Platform:** STM32MP1 / Linux