#pragma once
#include <string>
#include <vector>

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
};
