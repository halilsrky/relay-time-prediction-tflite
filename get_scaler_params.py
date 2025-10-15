import joblib
import numpy as np

# Scaler'ı yükle
scaler = joblib.load('tflite_models/scaler.pkl')

print("🔧 StandardScaler Parametreleri:")
print("================================")
print("Mean (ortalama) değerleri:")
for i, mean_val in enumerate(scaler.mean_):
    print(f"  Feature {i}: {mean_val:.6f}")

print("\nScale (standart sapma) değerleri:")
for i, scale_val in enumerate(scaler.scale_):
    print(f"  Feature {i}: {scale_val:.6f}")

print("\nC++ için hazır kod:")
print("mean_ = {", end="")
for i, mean_val in enumerate(scaler.mean_):
    if i > 0:
        print(", ", end="")
    print(f"{mean_val:.6f}f", end="")
print("};")

print("scale_ = {", end="")
for i, scale_val in enumerate(scaler.scale_):
    if i > 0:
        print(", ", end="")
    print(f"{scale_val:.6f}f", end="")
print("};")

# Feature isimleri de yazdır
feature_names = ['Ip', 'TMS', 'IL', 'Isc', 'PWT1', 'PWT3', 'FL', 'Ftype_2', 'Ftype_3', 'Ftype_4']
print("\nFeature isimleri:")
for i, name in enumerate(feature_names):
    print(f"  {i}: {name}")