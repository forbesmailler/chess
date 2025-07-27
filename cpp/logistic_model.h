#pragma once
#include <vector>
#include <string>

class LogisticModel {
public:
    LogisticModel();
    bool load_model(const std::string& model_path);
    std::vector<float> predict_proba(const std::vector<float>& features) const;
    
private:
    std::vector<float> weights;
    std::vector<float> intercept;
    int n_classes;
    int n_features;
    
    static float sigmoid(float x);
    static std::vector<float> softmax(const std::vector<float>& logits);
};
