#include <tensorflow/lite/c/c_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>

// Targets.csv dosyasını okuma fonksiyonu
std::vector<float> read_targets_csv(const std::string& filename) {
    std::vector<float> targets;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "❌ ERROR: " << filename << " dosyası açılamadı!" << std::endl;
        return targets;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        // İlk satırı atla (header: TIME)
        if (first_line) {
            first_line = false;
            continue;
        }
        
        // Boş satırları atla
        if (line.empty()) continue;
        
        try {
            float target = std::stof(line);
            targets.push_back(target);
        } catch (const std::exception& e) {
            std::cerr << "⚠️ Geçersiz satır atlandı: " << line << std::endl;
        }
    }
    
    file.close();
    std::cout << "✅ " << targets.size() << " target değeri okundu: " << filename << std::endl;
    return targets;
}

// Hata hesaplama ve raporlama fonksiyonu
void calculate_and_report_errors(const std::vector<float>& predictions, 
                                const std::vector<float>& targets) {
    if (predictions.size() != targets.size()) {
        std::cerr << "❌ Tahmin sayısı (" << predictions.size() 
                  << ") ile target sayısı (" << targets.size() << ") eşleşmiyor!" << std::endl;
        return;
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "📊 HATA ANALİZİ VE DOĞRULUK DEĞERLENDİRMESİ" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Hata hesaplamaları
    float sum_squared_error = 0.0f;
    float sum_absolute_error = 0.0f;
    float sum_targets = 0.0f;
    float sum_targets_squared = 0.0f;
    float max_error = 0.0f;
    float max_percentage_error = 0.0f;
    
    std::cout << "\n📋 Detaylı Hata Listesi:" << std::endl;
    std::cout << "Satır\tGerçek\tTahmin\tMutlak Hata\t%Hata" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        float target = targets[i];
        float prediction = predictions[i];
        float absolute_error = std::abs(target - prediction);
        float percentage_error = (std::abs(target - prediction) / target) * 100.0f;
        
        // Toplam hesaplar
        sum_squared_error += (target - prediction) * (target - prediction);
        sum_absolute_error += absolute_error;
        sum_targets += target;
        sum_targets_squared += target * target;
        
        // Maksimum hatalar
        if (absolute_error > max_error) {
            max_error = absolute_error;
        }
        if (percentage_error > max_percentage_error) {
            max_percentage_error = percentage_error;
        }
        
        // Her satırı yazdır
        std::cout << std::fixed << std::setprecision(4);
        std::cout << (i+1) << "\t" << target << "\t" << prediction << "\t" 
                  << absolute_error << "\t\t" << std::setprecision(1) 
                  << percentage_error << "%" << std::endl;
    }
    
    // Toplam istatistikler
    float n = static_cast<float>(predictions.size());
    float mse = sum_squared_error / n;
    float rmse = std::sqrt(mse);
    float mae = sum_absolute_error / n;
    float mean_target = sum_targets / n;
    
    // R² hesaplama
    float ss_tot = sum_targets_squared - (sum_targets * sum_targets) / n;
    float r_squared = 1.0f - (sum_squared_error / ss_tot);
    
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "📊 TOPLAM İSTATİSTİKLER:" << std::endl;
    std::cout << "   📏 Test Sayısı: " << static_cast<int>(n) << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "   🎯 R² (Açıklama Oranı): " << r_squared << std::endl;
    std::cout << "   📐 RMSE (Kök Ortalama Kare Hata): " << rmse << " saniye" << std::endl;
    std::cout << "   📊 MAE (Ortalama Mutlak Hata): " << mae << " saniye" << std::endl;
    std::cout << "   ⚠️ Maksimum Mutlak Hata: " << max_error << " saniye" << std::endl;
    std::cout << "   📈 Maksimum %Hata: " << std::setprecision(1) << max_percentage_error << "%" << std::endl;
    std::cout << "   🎲 Ortalama Target Değeri: " << std::setprecision(4) << mean_target << " saniye" << std::endl;
    
    // Kalite değerlendirmesi
    std::cout << "\n🏆 MODEL KALİTE DEĞERLENDİRMESİ:" << std::endl;
    if (r_squared > 0.999) {
        std::cout << "   ✅ MÜKEMMEL: R² > 0.999 - Üretim kalitesi" << std::endl;
    } else if (r_squared > 0.99) {
        std::cout << "   ✅ ÇOK İYİ: R² > 0.99 - Kabul edilebilir kalite" << std::endl;
    } else if (r_squared > 0.95) {
        std::cout << "   ⚠️ ORTA: R² > 0.95 - Geliştirme gerekli" << std::endl;
    } else {
        std::cout << "   ❌ DÜŞÜK: R² < 0.95 - Model revizyon gerekiyor" << std::endl;
    }
    
    if (max_percentage_error < 5.0f) {
        std::cout << "   ✅ Maksimum hata < %5 - Güvenlik açısından uygun" << std::endl;
    } else if (max_percentage_error < 10.0f) {
        std::cout << "   ⚠️ Maksimum hata < %10 - Dikkatli kullanım önerilir" << std::endl;
    } else {
        std::cout << "   ❌ Maksimum hata > %10 - Güvenlik riski var!" << std::endl;
    }
}

class TFLiteXGBoostEquivalent {
private:
    TfLiteModel* model_;
    TfLiteInterpreter* interpreter_;
    
public:
    TFLiteXGBoostEquivalent(const std::string& model_path) {
        // Model dosyasını yükle
        model_ = TfLiteModelCreateFromFile(model_path.c_str());
        if (!model_) {
            std::cerr << "❌ Model yüklenemedi: " << model_path << std::endl;
            exit(1);
        }
        
        // Interpreter oluştur
        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        interpreter_ = TfLiteInterpreterCreate(model_, options);
        TfLiteInterpreterOptionsDelete(options);
        
        if (!interpreter_) {
            std::cerr << "❌ Interpreter oluşturulamadı!" << std::endl;
            exit(1);
        }
        
        // Tensörleri allocate et
        if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) {
            std::cerr << "❌ Tensörler allocate edilemedi!" << std::endl;
            exit(1);
        }
        
        std::cout << "✅ XGBoost-M2Cgen modeli yüklendi: " << model_path << std::endl;
    }
    
    ~TFLiteXGBoostEquivalent() {
        TfLiteInterpreterDelete(interpreter_);
        TfLiteModelDelete(model_);
    }
    
    float predict(const std::vector<float>& features) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Input tensörü al
        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter_, 0);
        if (!input_tensor) {
            std::cerr << "❌ Input tensör alınamadı!" << std::endl;
            return -1.0f;
        }
        
        // Input verisini tensöre kopyala (Scaling YOK!)
        float* input_data = reinterpret_cast<float*>(TfLiteTensorData(input_tensor));
        for (size_t i = 0; i < features.size(); ++i) {
            input_data[i] = features[i];
        }
        
        // Inference çalıştır
        if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk) {
            std::cerr << "❌ Inference çalıştırılamadı!" << std::endl;
            return -1.0f;
        }
        
        // Output tensörü al
        const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
        if (!output_tensor) {
            std::cerr << "❌ Output tensör alınamadı!" << std::endl;
            return -1.0f;
        }
        
        // Sonucu al
        const float* output_data = reinterpret_cast<const float*>(TfLiteTensorData(output_tensor));
        float prediction = output_data[0];
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "⏱️ Tahmin süresi: " << duration.count() << " µs" << std::endl;
        
        return prediction;
    }
};

std::vector<std::vector<float>> readCSV(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "❌ CSV dosyası açılamadı: " << filename << std::endl;
        return data;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false; // Header'ı atla
            continue;
        }
        
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stof(cell));
            } catch (const std::exception& e) {
                std::cerr << "❌ Sayı dönüştürme hatası: " << cell << std::endl;
            }
        }
        
        if (row.size() == 10) { // 10 feature bekliyoruz
            data.push_back(row);
        }
    }
    
    file.close();
    return data;
}

int main() {
    std::cout << "🚀 XGBoost-M2Cgen TensorFlow Lite C++ Test Başlatılıyor..." << std::endl;
    std::cout << "================================================================" << std::endl;
    
    // Model ve CSV dosya yolları
    const std::string model_path = "tflite_models/xgboost_m2cgen.tflite";
    const std::string csv_path = "input.csv";
    const std::string targets_path = "targets.csv";
    
    // XGBoost-M2Cgen modelini yükle
    TFLiteXGBoostEquivalent xgb_nn(model_path);
    
    // CSV dosyasını oku
    std::cout << "\n📊 CSV dosyası okunuyor: " << csv_path << std::endl;
    auto input_data = readCSV(csv_path);
    
    if (input_data.empty()) {
        std::cerr << "❌ CSV dosyasında veri bulunamadı!" << std::endl;
        return 1;
    }
    
    // Targets dosyasını oku
    std::vector<float> targets = read_targets_csv(targets_path);
    if (targets.empty()) {
        std::cout << "⚠️ Targets dosyası bulunamadı, sadece tahminler gösterilecek." << std::endl;
    }
    
    std::cout << "✅ " << input_data.size() << " satır veri okundu" << std::endl;
    
    // Feature isimleri
    std::vector<std::string> feature_names = {
        "Ip", "TMS", "IL", "Isc", "PWT1", "PWT3", "FL", "Ftype_2", "Ftype_3", "Ftype_4"
    };
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🌲 XGBOOST-M2CGEN TAHMİNLERİ (Scaling YOK)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Tahminleri saklamak için vector
    std::vector<float> predictions;
    
    // Her satır için tahmin yap
    for (size_t i = 0; i < input_data.size(); ++i) {
        std::cout << "\n📋 Satır " << (i + 1) << ":" << std::endl;
        
        // Input feature'ları göster
        std::cout << "   📥 Input Features:" << std::endl;
        for (size_t j = 0; j < input_data[i].size(); ++j) {
            std::cout << "      " << std::left << std::setw(8) << feature_names[j] 
                      << ": " << std::fixed << std::setprecision(2) << input_data[i][j] << std::endl;
        }
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Tahmin yap
        float prediction = xgb_nn.predict(input_data[i]);
        predictions.push_back(prediction);
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        
        std::cout << "   🎯 Tahmin Edilen TIME: " << std::fixed << std::setprecision(4) 
                  << prediction << " saniye" << std::endl;
        
        // Eğer target değeri varsa göster
        if (!targets.empty() && i < targets.size()) {
            float target = targets[i];
            float error = std::abs(target - prediction);
            float percentage_error = (error / target) * 100.0f;
            std::cout << "   🎲 Gerçek TIME: " << target << " saniye" << std::endl;
            std::cout << "   📊 Mutlak Hata: " << error << " saniye (" << std::setprecision(1) << percentage_error << "%)" << std::endl;
        }
        
        std::cout << "   ⏱️ Toplam İşlem Süresi: " << total_duration.count() << " µs ("
                  << std::fixed << std::setprecision(3) << total_duration.count() / 1000.0 << " ms)" << std::endl;
        
        // Ftype bilgisini decode et
        std::string ftype = "1"; // Default
        if (input_data[i][7] == 1) ftype = "2";
        else if (input_data[i][8] == 1) ftype = "3";
        else if (input_data[i][9] == 1) ftype = "4";
        
        std::cout << "   📊 Arıza Tipi (Ftype): " << ftype << std::endl;
        std::cout << "   " << std::string(50, '-') << std::endl;
    }
    
    std::cout << "\n🎉 Tüm tahminler tamamlandı!" << std::endl;
    std::cout << "✅ XGBoost-M2Cgen modeli başarıyla test edildi." << std::endl;
    
    // Eğer targets varsa hata analizi yap
    if (!targets.empty() && targets.size() == predictions.size()) {
        calculate_and_report_errors(predictions, targets);
    } else if (!targets.empty()) {
        std::cout << "⚠️ Target sayısı (" << targets.size() << ") ile tahmin sayısı (" 
                  << predictions.size() << ") eşleşmiyor, hata analizi yapılamadı." << std::endl;
    }
    
    return 0;
}