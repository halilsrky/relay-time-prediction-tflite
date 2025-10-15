
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

print("\nÇoklu tahmin örneği:")
for i, features in enumerate(test_cases):
    pred = predict_time(features)
    print(f"Test {i+1}: {pred:.4f} saniye")
