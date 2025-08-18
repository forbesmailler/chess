#include "logistic_model.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

LogisticModel::LogisticModel() : n_classes(3), n_features(1542) {}

bool LogisticModel::load_model(const std::string& model_path) {
    std::string txt_path = model_path;
    if (txt_path.find(".joblib") != std::string::npos) {
        txt_path = txt_path.substr(0, txt_path.find(".joblib")) + "_coefficients.txt";
    }
    
    std::ifstream file(txt_path);
    if (!file.is_open()) {
        file.open(model_path);
        if (!file.is_open()) {
            std::cerr << "Could not open model file: " << model_path << " or " << txt_path << std::endl;
            std::cerr << "Please run export_model.py in the cpp/train/ directory to convert your .joblib model to text format" << std::endl;
            
            weights.resize(n_features * n_classes, 0.001f);
            intercept.resize(n_classes, 0.0f);
            
            for (size_t i = 0; i < weights.size(); i++) {
                weights[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.001f;
            }
            
            std::cout << "Using dummy model (model file not found)" << std::endl;
            return false;
        }
    }
    
    std::string line;
    bool reading_intercept = false;
    bool reading_coefficients = false;
    
    intercept.clear();
    weights.clear();
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        if (line == "INTERCEPT") {
            reading_intercept = true;
            reading_coefficients = false;
            continue;
        } else if (line == "COEFFICIENTS") {
            reading_intercept = false;
            reading_coefficients = true;
            continue;
        }
        
        if (reading_intercept) {
            intercept.push_back(std::stof(line));
        } else if (reading_coefficients) {
            weights.push_back(std::stof(line));
        }
    }
    
    file.close();
    
    n_classes = static_cast<int>(intercept.size());
    if (!weights.empty() && n_classes > 0) {
        n_features = static_cast<int>(weights.size() / n_classes);
    }
    
    std::cout << "Loaded model with " << n_classes << " classes and " << n_features << " features" << std::endl;
    return !weights.empty() && !intercept.empty();
}

std::vector<float> LogisticModel::predict_proba(const std::vector<float>& features) const {
    if (features.size() != static_cast<size_t>(n_features)) {
        std::cerr << "Feature size mismatch: expected " << n_features 
                  << ", got " << features.size() << std::endl;
        return {0.33f, 0.34f, 0.33f};
    }
    
    std::vector<float> logits(n_classes, 0.0f);
    for (int class_idx = 0; class_idx < n_classes; class_idx++) {
        logits[class_idx] = intercept[class_idx];
        for (int feature_idx = 0; feature_idx < n_features; feature_idx++) {
            logits[class_idx] += weights[class_idx * n_features + feature_idx] * features[feature_idx];
        }
    }
    
    return softmax(logits);
}

float LogisticModel::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> LogisticModel::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }
    
    return probs;
}
