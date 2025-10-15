"""
Neural Network Modelini TensorFlow Lite (.tflite) Formatına Dönüştürme
Neural Network modelini eğitip .tflite formatında kaydeder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TFLiteModelConverter:
    """Neural Network modelini TensorFlow Lite formatına dönüştürme sınıfı"""
    
    def __init__(self, train_file='train.xlsx', test_file='test.xlsx'):
        self.train_file = train_file
        self.test_file = test_file
        self.output_dir = 'tflite_models'
        
        # Sonuçları saklamak için
        self.results = {}
        self.models = {}
        self.scaler = StandardScaler()
        
        # Output klasörü oluştur
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print("🚀 TensorFlow Lite Neural Network Dönüştürücü Başlatılıyor...")
        
    def load_and_prepare_data(self):
        """Veri yükleme ve ön işleme"""
        print("\n📊 Veriler yükleniyor...")
        
        # Verileri yükle
        self.train_df = pd.read_excel(self.train_file)
        self.test_df = pd.read_excel(self.test_file)
        
        print(f"✅ Train verisi: {self.train_df.shape}")
        print(f"✅ Test verisi: {self.test_df.shape}")
        
        # Ftype dağılımını göster
        print(f"\n📈 Ftype Dağılımı (Train):")
        ftype_counts = self.train_df['Ftype'].value_counts().sort_index()
        for ftype, count in ftype_counts.items():
            percentage = (count / len(self.train_df)) * 100
            print(f"   Ftype {ftype}: {count:,} adet ({percentage:.1f}%)")
        
        # One-hot encoding uygula
        print("\n🔧 One-hot encoding uygulanıyor...")
        
        # Train ve test için ayrı ayrı one-hot encoding
        train_features = self.train_df.drop(['TIME'], axis=1)
        test_features = self.test_df.drop(['TIME'], axis=1)
        
        # One-hot encoding (Ftype için)
        train_encoded = pd.get_dummies(train_features, columns=['Ftype'], prefix='Ftype', drop_first=True)
        test_encoded = pd.get_dummies(test_features, columns=['Ftype'], prefix='Ftype', drop_first=True)
        
        # Test setinde eksik olan sütunları ekle
        for col in train_encoded.columns:
            if col not in test_encoded.columns:
                test_encoded[col] = 0
                
        # Sütun sırasını aynı yap
        test_encoded = test_encoded[train_encoded.columns]
        
        # X ve y değişkenlerini ayır
        self.X_train = train_encoded
        self.X_test = test_encoded
        self.y_train = self.train_df['TIME']
        self.y_test = self.test_df['TIME']
        
        # Scaling (Neural Network için)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   Önceki feature sayısı: {len(train_features.columns)}")
        print(f"   Sonraki feature sayısı: {len(self.X_train.columns)}")
        print(f"   Yeni feature isimleri: {list(self.X_train.columns)}")
        print("✅ Veri hazırlığı tamamlandı!")
        
    def train_tensorflow_neural_network(self):
        """TensorFlow/Keras ile Neural Network eğitimi"""
        print("\n🧠 TensorFlow Neural Network Eğitiliyor...")
        
        # Model mimarisi (sklearn MLPRegressor'a benzer)
        model = keras.Sequential([
            layers.Dense(100, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(50, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Eğitim
        history = model.fit(
            self.X_train_scaled, self.y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Tahminler
        y_pred_train = model.predict(self.X_train_scaled, verbose=0).flatten()
        y_pred_test = model.predict(self.X_test_scaled, verbose=0).flatten()
        
        # Metrikler
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_r2 = r2_score(self.y_train, y_pred_train)
        
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        overfitting = train_r2 - test_r2
        
        print(f"   Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"   RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        print(f"   Eğitim durdu: {len(history.history['loss'])} epoch")
        print(f"   Overfitting: {overfitting:+.4f}")
        
        # Sonuçları kaydet
        self.results['Neural_Network'] = {
            'RMSE': test_rmse,
            'MAE': test_mae,
            'R²': test_r2,
            'Train_R²': train_r2,
            'Overfitting_Score': overfitting
        }
        
        self.models['Neural_Network'] = model
        return model, history
        
    def convert_to_tflite(self, tf_model, model_name):
        """TensorFlow modelini TFLite formatına dönüştür"""
        print(f"\n🔄 {model_name} TensorFlow Lite formatına dönüştürülüyor...")
        
        # TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        
        # Optimizasyonlar
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Dönüştürme
        tflite_model = converter.convert()
        
        # Dosyaya kaydet
        tflite_path = f"{self.output_dir}/{model_name}_model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        # Model boyutunu kontrol et
        model_size = os.path.getsize(tflite_path) / 1024  # KB
        print(f"✅ {model_name} TFLite modeli kaydedildi: {tflite_path}")
        print(f"   Model boyutu: {model_size:.1f} KB")
        
        return tflite_path, model_size
        
    def test_tflite_models(self):
        """TFLite modellerini test et"""
        print("\n🧪 TensorFlow Lite modeli test ediliyor...")
        
        # Neural Network TFLite test
        nn_tflite_path = f"{self.output_dir}/tensorflow_nn_model.tflite"
        if os.path.exists(nn_tflite_path):
            interpreter = tf.lite.Interpreter(model_path=nn_tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test örneği
            test_sample = self.X_test_scaled[:5].astype(np.float32)
            predictions = []
            
            for sample in test_sample:
                interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1))
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
                predictions.append(pred)
                
            print(f"✅ Neural Network TFLite test başarılı!")
            print(f"   Test örnekleri: {len(predictions)}")
            print(f"   Örnek tahminler: {predictions[:3]}")
        else:
            print("❌ Neural Network TFLite modeli bulunamadı!")
            
    def save_supporting_files(self):
        """Destekleyici dosyaları kaydet"""
        print("\n💾 Destekleyici dosyalar kaydediliyor...")
        
        # Scaler'ı kaydet
        joblib.dump(self.scaler, f'{self.output_dir}/scaler.pkl')
        print("✅ StandardScaler kaydedildi")
        
        # Feature isimleri kaydet
        feature_names = list(self.X_train.columns)
        with open(f'{self.output_dir}/feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print("✅ Feature isimleri kaydedildi")
        
        # Model sonuçları kaydet
        results_df = pd.DataFrame(self.results).T
        results_df.to_excel(f'{self.output_dir}/model_results.xlsx')
        print("✅ Model sonuçları kaydedildi")
        
    def create_usage_example(self):
        """Kullanım örneği kodu oluştur"""
        usage_code = '''
# TensorFlow Lite Neural Network Model Kullanım Örneği

import tensorflow as tf
import numpy as np
import joblib

# 1. TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path="tflite_models/tensorflow_nn_model.tflite")
interpreter.allocate_tensors()

# Input/Output detayları
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Scaler'ı yükle (Neural Network için)
scaler = joblib.load("tflite_models/scaler.pkl")

# 3. Tahmin yapma fonksiyonu
def predict_time(features):
    """
    features: [Ip, TMS, IL, Isc, PWT1, PWT3, FL, Ftype_2, Ftype_3, Ftype_4]
    """
    # Özellikleri numpy array'e çevir
    input_data = np.array(features, dtype=np.float32).reshape(1, -1)
    
    # Scaling uygula (Neural Network için)
    input_scaled = scaler.transform(input_data).astype(np.float32)
    
    # TFLite modeli ile tahmin
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    return prediction

# 4. Örnek kullanım
example_features = [110.58, 0.1, 92.15, 5893.56, 2.5, 2.5, 1.0, 0, 0, 0]  # Ftype=1
predicted_time = predict_time(example_features)
print(f"Tahmin edilen TIME: {predicted_time:.4f} saniye")

# 5. Çoklu tahmin örneği
test_cases = [
    [110.58, 0.1, 92.15, 5893.56, 2.5, 2.5, 1.0, 0, 0, 0],  # Ftype=1
    [98.24, 0.05, 82.45, 4521.33, 1.5, 1.5, 0.5, 1, 0, 0],  # Ftype=2
    [125.75, 0.2, 105.60, 7234.89, 3.0, 3.0, 2.0, 0, 1, 0]  # Ftype=3
]

print("\\nÇoklu tahmin örneği:")
for i, features in enumerate(test_cases):
    pred = predict_time(features)
    print(f"Test {i+1}: {pred:.4f} saniye")
'''
        
        with open(f'{self.output_dir}/usage_example.py', 'w') as f:
            f.write(usage_code)
        print("✅ Kullanım örneği kaydedildi")
        
    def run_conversion(self):
        """Ana dönüştürme işlemini çalıştır"""
        start_time = datetime.now()
        
        # 1. Veri hazırlama
        self.load_and_prepare_data()
        
        # 2. TensorFlow Neural Network eğitimi
        tf_nn_model, history = self.train_tensorflow_neural_network()
        
        # 3. TensorFlow modelini TFLite'a dönüştür
        nn_tflite_path, nn_size = self.convert_to_tflite(tf_nn_model, "tensorflow_nn")
        
        # 4. TFLite modelini test et
        self.test_tflite_models()
        
        # 5. Destekleyici dosyaları kaydet
        self.save_supporting_files()
        
        # 6. Kullanım örneği oluştur
        self.create_usage_example()
        
        # Özet
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("🎉 NEURAL NETWORK TensorFlow Lite DÖNÜŞTÜRME TAMAMLANDI!")
        print("="*60)
        print("📁 Oluşturulan dosyalar:")
        print(f"   - {self.output_dir}/tensorflow_nn_model.tflite ({nn_size:.1f} KB)")
        print(f"   - {self.output_dir}/scaler.pkl (StandardScaler)")
        print(f"   - {self.output_dir}/feature_names.txt (Feature isimleri)")
        print(f"   - {self.output_dir}/model_results.xlsx (Sonuçlar)")
        print(f"   - {self.output_dir}/usage_example.py (Kullanım örneği)")
        
        print(f"\n🎯 SONUÇLAR:")
        results_df = pd.DataFrame(self.results).T
        print(results_df[['R²', 'RMSE', 'MAE']].round(4))
        
        print(f"\n⏱️ Toplam süre: {duration:.1f} saniye")
        print(f"✅ Neural Network modeli embedded sistemlerde kullanıma hazır!")


def main():
    """Ana fonksiyon"""
    converter = TFLiteModelConverter()
    converter.run_conversion()


if __name__ == "__main__":
    main()