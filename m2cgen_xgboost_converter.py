#!/usr/bin/env python3
"""
M2Cgen ile XGBoost → TensorFlow Lite Dönüştürücü
================================================
M2Cgen kullanarak XGBoost modelini önce kod formatına çevirip
sonra TensorFlow Lite'a dönüştürür.
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

def install_m2cgen():
    """M2Cgen paketini yükle"""
    import subprocess
    import sys
    
    try:
        import m2cgen as m2c
        print("✅ M2Cgen zaten yüklü")
        return True
    except ImportError:
        print("📦 M2Cgen yükleniyor...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "m2cgen"])
            import m2cgen as m2c
            print("✅ M2Cgen başarıyla yüklendi")
            return True
        except Exception as e:
            print(f"❌ M2Cgen yüklenemedi: {e}")
            return False

def prepare_data():
    """Veri setini hazırla"""
    print("📊 Veri seti yükleniyor...")
    
    # Excel dosyasını oku
    df = pd.read_excel('train.xlsx')
    print(f"✅ {len(df)} satır veri yüklendi")
    
    # Feature engineering
    feature_columns = ['Ip', 'TMS', 'IL', 'Isc', 'PWT1', 'PWT3', 'FL']
    
    # Ftype sütununu one-hot encoding yap
    ftype_dummies = pd.get_dummies(df['Ftype'], prefix='Ftype')
    
    # Ana features ile birleştir
    X = pd.concat([df[feature_columns], ftype_dummies], axis=1)
    y = df['TIME']
    
    print(f"📋 Feature sayısı: {X.shape[1]}")
    print(f"📋 Feature isimleri: {list(X.columns)}")
    
    return X, y

def train_xgboost(X, y):
    """XGBoost model eğit"""
    print("\n🌲 XGBoost model eğitiliyor...")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # XGBoost parametreleri
    xgb_params = {
        'n_estimators': 50,  # Daha az ağaç (TFLite için)
        'max_depth': 4,      # Daha sığ ağaçlar
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'auto'
    }
    
    # Model eğit
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    
    # Test et
    y_pred = xgb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"✅ XGBoost R² Score: {r2:.6f}")
    print(f"✅ XGBoost MSE: {mse:.6f}")
    
    return xgb_model, X_train, X_test, y_train, y_test

def convert_with_m2cgen(xgb_model, feature_names):
    """M2Cgen ile XGBoost'u Python koduna çevir"""
    print("\n🔄 M2Cgen ile XGBoost → Python kodu dönüşümü...")
    
    try:
        import m2cgen as m2c
        
        # XGBoost'u Python koduna çevir
        code = m2c.export_to_python(xgb_model, function_name="xgb_predict")
        
        # Kodu dosyaya kaydet
        with open('xgboost_m2cgen.py', 'w') as f:
            f.write(code)
        
        print("✅ XGBoost Python kodu oluşturuldu: xgboost_m2cgen.py")
        
        # Kodun bir kısmını göster
        lines = code.split('\n')
        print("\n📄 Oluşturulan kod örneği:")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1:2d}: {line}")
        print("    ...")
        
        return code
        
    except Exception as e:
        print(f"❌ M2Cgen dönüşüm hatası: {e}")
        return None

def create_tensorflow_wrapper(xgb_model, X_sample, feature_names):
    """XGBoost'u taklit eden TensorFlow modeli oluştur"""
    print("\n🧠 TensorFlow wrapper modeli oluşturuluyor...")
    
    # Daha fazla synthetic data oluştur
    n_samples = 50000
    
    # Feature'ları ayrı ayrı örnekle
    X_synthetic_list = []
    for col_idx in range(X_sample.shape[1]):
        col_data = X_sample.iloc[:, col_idx]
        if col_data.nunique() <= 2:  # Binary feature (Ftype)
            synthetic_col = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        else:  # Continuous feature
            mean_val = col_data.mean()
            std_val = col_data.std()
            synthetic_col = np.random.normal(mean_val, std_val, n_samples)
            # Minimum değerleri sıfırdan küçük olmasın
            synthetic_col = np.maximum(synthetic_col, 0)
    
        X_synthetic_list.append(synthetic_col)
    
    X_synthetic = np.column_stack(X_synthetic_list).astype(np.float32)
    
    # XGBoost tahminleri al
    y_synthetic = xgb_model.predict(X_synthetic).astype(np.float32)
    
    print(f"📊 Synthetic veri: {X_synthetic.shape}")
    print(f"📊 Y aralığı: {y_synthetic.min():.4f} - {y_synthetic.max():.4f}")
    
    # TensorFlow modeli oluştur - XGBoost'a daha yakın
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_sample.shape[1],), name='input'),
        
        # İlk katman - feature extraction
        tf.keras.layers.Dense(256, activation='relu', name='dense1'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        # İkinci katman - complex patterns
        tf.keras.layers.Dense(128, activation='relu', name='dense2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        # Üçüncü katman - refinement
        tf.keras.layers.Dense(64, activation='relu', name='dense3'),
        tf.keras.layers.Dropout(0.05),
        
        # Son katman
        tf.keras.layers.Dense(1, activation='linear', name='output')
    ])
    
    # Modeli derle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Modeli eğit
    print("🎯 TensorFlow modeli XGBoost'u taklit ediyor...")
    history = model.fit(
        X_synthetic, y_synthetic,
        epochs=200,
        batch_size=256,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
        ]
    )
    
    # Test et
    X_sample_array = X_sample.values.astype(np.float32)
    y_pred_tf = model.predict(X_sample_array, verbose=0).flatten()
    y_pred_xgb = xgb_model.predict(X_sample)
    
    # Korelasyon hesapla
    correlation = np.corrcoef(y_pred_tf, y_pred_xgb)[0, 1]
    mse_diff = mean_squared_error(y_pred_xgb, y_pred_tf)
    
    print(f"✅ TensorFlow-XGBoost korelasyonu: {correlation:.6f}")
    print(f"✅ TensorFlow-XGBoost MSE: {mse_diff:.6f}")
    
    # Örnek tahminler
    print("\n📊 Örnek Tahminler Karşılaştırması:")
    print("XGBoost\tTensorFlow\tFark")
    print("-" * 35)
    for i in range(min(5, len(y_pred_tf))):
        diff = abs(y_pred_xgb[i] - y_pred_tf[i])
        print(f"{y_pred_xgb[i]:.4f}\t{y_pred_tf[i]:.4f}\t\t{diff:.4f}")
    
    return model

def convert_to_tflite(tf_model, X_sample):
    """TensorFlow modelini TFLite'a çevir"""
    print("\n🔄 TensorFlow → TensorFlow Lite dönüşümü...")
    
    # TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    
    # Optimizasyonlar
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    # Representative dataset
    def representative_dataset():
        for i in range(100):
            sample = X_sample.iloc[i:i+1].values.astype(np.float32)
            yield [sample]
    
    converter.representative_dataset = representative_dataset
    
    # Dönüştür
    try:
        tflite_model = converter.convert()
        
        # Kaydet
        os.makedirs('tflite_models', exist_ok=True)
        tflite_path = 'tflite_models/xgboost_m2cgen.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model kaydedildi: {tflite_path}")
        print(f"📦 Model boyutu: {len(tflite_model) / 1024:.1f} KB")
        
        return tflite_path, tflite_model
        
    except Exception as e:
        print(f"❌ TFLite dönüşüm hatası: {e}")
        return None, None

def test_tflite_model(tflite_path, X_test, y_test, xgb_model):
    """TFLite modelini test et"""
    print("\n🧪 TFLite model test ediliyor...")
    
    try:
        # TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Input/output detayları
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📋 Input shape: {input_details[0]['shape']}")
        print(f"📋 Output shape: {output_details[0]['shape']}")
        print(f"📋 Input dtype: {input_details[0]['dtype']}")
        print(f"📋 Output dtype: {output_details[0]['dtype']}")
        
        # Test tahminleri
        n_test = min(100, len(X_test))
        tflite_predictions = []
        xgb_predictions = []
        
        for i in range(n_test):
            # TFLite tahmin
            input_data = X_test.iloc[i:i+1].values.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            tflite_pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
            tflite_predictions.append(tflite_pred)
            
            # XGBoost tahmin
            xgb_pred = xgb_model.predict(X_test.iloc[i:i+1])[0]
            xgb_predictions.append(xgb_pred)
        
        # Karşılaştır
        tflite_predictions = np.array(tflite_predictions)
        xgb_predictions = np.array(xgb_predictions)
        
        # Metrikleri hesapla
        correlation = np.corrcoef(tflite_predictions, xgb_predictions)[0, 1]
        mse = mean_squared_error(xgb_predictions, tflite_predictions)
        mae = np.mean(np.abs(xgb_predictions - tflite_predictions))
        max_error = np.max(np.abs(xgb_predictions - tflite_predictions))
        
        print(f"✅ TFLite-XGBoost korelasyonu: {correlation:.6f}")
        print(f"✅ TFLite-XGBoost MSE: {mse:.6f}")
        print(f"✅ TFLite-XGBoost MAE: {mae:.6f}")
        print(f"✅ TFLite-XGBoost Max Error: {max_error:.6f}")
        
        # Örnek tahminler göster
        print("\n📊 Detaylı Tahmin Karşılaştırması:")
        print("XGBoost\tTFLite\tFark\t%Fark")
        print("-" * 40)
        for i in range(min(10, len(tflite_predictions))):
            diff = abs(xgb_predictions[i] - tflite_predictions[i])
            pct_diff = (diff / xgb_predictions[i]) * 100
            print(f"{xgb_predictions[i]:.4f}\t{tflite_predictions[i]:.4f}\t{diff:.4f}\t{pct_diff:.1f}%")
        
        return correlation > 0.95  # %95 korelasyon başarı kabul edilir
        
    except Exception as e:
        print(f"❌ TFLite test hatası: {e}")
        return False

def main():
    """Ana fonksiyon"""
    print("🚀 M2Cgen ile XGBoost → TFLite Dönüştürücü Başlıyor...")
    print("=" * 70)
    
    try:
        # 1. M2Cgen'i yükle
        if not install_m2cgen():
            print("❌ M2Cgen yüklenemedi, çıkılıyor...")
            return
        
        # 2. Veri hazırla
        X, y = prepare_data()
        
        # 3. XGBoost model eğit
        xgb_model, X_train, X_test, y_train, y_test = train_xgboost(X, y)
        
        # 4. XGBoost'u kaydet
        os.makedirs('tflite_models', exist_ok=True)
        joblib.dump(xgb_model, 'tflite_models/xgboost_m2cgen_model.pkl')
        print("✅ XGBoost model kaydedildi: tflite_models/xgboost_m2cgen_model.pkl")
        
        # 5. M2Cgen ile koda çevir
        code = convert_with_m2cgen(xgb_model, X.columns.tolist())
        
        # 6. TensorFlow wrapper oluştur
        tf_model = create_tensorflow_wrapper(xgb_model, X_train, X.columns.tolist())
        
        # 7. TFLite'a çevir
        tflite_path, tflite_model = convert_to_tflite(tf_model, X_train)
        
        if tflite_path:
            # 8. Test et
            test_success = test_tflite_model(tflite_path, X_test, y_test, xgb_model)
            
            if test_success:
                print("\n🎉 M2Cgen dönüşümü başarıyla tamamlandı!")
                print(f"📁 TFLite model: {tflite_path}")
                print(f"📁 Python kodu: xgboost_m2cgen.py")
                print("🔧 Artık C++ kodunuzda bu modeli kullanabilirsiniz.")
                
                # Ana dizine de kopyala
                import shutil
                shutil.copy(tflite_path, 'xgboost_m2cgen.tflite')
                print("✅ Model ana dizine kopyalandı: xgboost_m2cgen.tflite")
                
            else:
                print("\n⚠️ Dönüşüm tamamlandı ama test başarısız!")
                print("Model doğruluğu yeterli değil.")
        else:
            print("\n❌ TFLite dönüşümü başarısız!")
            
    except Exception as e:
        print(f"\n❌ Ana hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()