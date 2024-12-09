
# **Sentiment Analysis of Rotten Tomatoes Movie Reviews**

### **Team Members**  
**Shicheng Yan**  
**Chengbin Huang**

---

## **Abstract**

This study evaluates the performance of Transformer-based models—BERT and GPT-2—for sentiment analysis using the Rotten Tomatoes movie review dataset. By experimenting with data cleaning techniques and class-weighted optimizations, the project highlights the trade-offs between noise reduction, class balance, and model performance. Results reveal that BERT, when optimized with class weighting on uncleaned data, achieves the highest overall performance, demonstrating the importance of preserving critical features and addressing class imbalances.

---

## **1. Introduction**

### **1.1 Problem Statement**
The goal of this project is to classify movie review phrases into one of five sentiment categories: Negative, Somewhat Negative, Neutral, Somewhat Positive, and Positive. Sentiment analysis is widely applied in domains such as recommendation systems, social media analytics, and market research.

### **1.2 Motivation**
Understanding sentiment is pivotal for tailoring user experiences, analyzing public opinion, and generating actionable insights in industries like e-commerce, entertainment, and customer service. Despite the significance, challenges such as noisy data, imbalanced class distributions, and task-specific model selection remain open research questions.

---

## **2. Dataset Overview**

### **2.1 Source**
The dataset comprises movie reviews parsed into phrases, each labeled with a sentiment score. It contains approximately 156,000 phrases.

### **2.2 Class Distribution**
- **Neutral**: 51% of samples, the largest category.
- **Positive and Negative**: Highly underrepresented at ~4% and ~10%, respectively.
- **Imbalance Challenge**: The dominance of Neutral reviews skews model predictions towards overfitting on this class.

---

## **3. Methodology**

### **3.1 Data Preprocessing**
1. **Noise Reduction**:
   - Removed punctuation, special characters, and stopwords.
   - Filtered short neutral phrases lacking sufficient sentiment context.
2. **Trade-offs**:
   - While cleaning reduced noise, it inadvertently removed features crucial for understanding context, particularly in the Neutral class.

### **3.2 Models**
1. **BERT**:
   - A bidirectional Transformer designed to capture contextual dependencies in textual data.
   - Fine-tuned for sentiment classification.
2. **GPT-2**:
   - A generative Transformer model repurposed for classification.
   - Required padding adjustments due to its design limitations for batch training.

### **3.3 Class-Weighted Optimization**
To address class imbalance, we applied weighted loss functions during training. This adjustment penalized misclassifications in minority classes, particularly Positive and Negative sentiments.

### **3.4 Evaluation Metrics**
1. **Accuracy**: Measures overall correctness.
2. **F1-Scores**:
   - Macro-F1: Treats all classes equally.
   - Weighted F1: Adjusts scores based on class sizes.

---

## **4. Experimental Results**

### **4.1 Performance Summary**

| Model              | Dataset    | Accuracy | Macro F1 | Weighted F1 |
|--------------------|------------|----------|----------|-------------|
| BERT               | Cleaned    | 63.93%   | 60%      | 64%         |
| GPT-2              | Cleaned    | 61.14%   | 56%      | 61%         |
| BERT               | Uncleaned  | 69.80%   | 63%      | 70%         |
| GPT-2              | Uncleaned  | 68.75%   | 59%      | 69%         |
| BERT (Weighted)    | Uncleaned  | 70.04%   | 63%      | 70%         |

### **4.2 Observations**
- **Data Cleaning**:
  - Cleaning improved precision for nuanced categories like Somewhat Negative and Somewhat Positive but reduced overall accuracy due to the loss of Neutral context.
- **Class Weighting**:
  - Applying class-weighted loss significantly boosted BERT’s ability to handle minority classes, increasing its Macro-F1 score.
- **Model Performance**:
  - BERT consistently outperformed GPT-2, indicating its superior ability to capture contextual relationships in shorter text fragments.

---

## **5. Discussion**

### **5.1 Key Takeaways**
1. **Cleaning Trade-offs**:
   - While noise reduction helps, excessive cleaning risks losing valuable context, particularly for Neutral samples.
2. **Class Balancing**:
   - Weighted loss functions improved minority class performance without degrading overall metrics.
3. **Model Suitability**:
   - BERT’s bidirectional context encoding made it better suited for the classification task than GPT-2, which is generative by design.

### **5.2 Limitations**
1. **Data Imbalance**:
   - Positive and Negative samples were underrepresented, limiting model generalization.
2. **Task-Specific Fine-Tuning**:
   - GPT-2’s architecture requires additional optimization for shorter and imbalanced text data.

---

## **6. Conclusion**

### **Summary**
This project demonstrates the importance of carefully balancing data preprocessing with model optimization in sentiment analysis tasks. The experiments reveal:
- BERT, with class-weighted optimization on uncleaned data, achieved the highest performance metrics.
- Data cleaning should be approached cautiously to avoid losing sentiment-relevant features.

### **Future Directions**
1. **Semi-Supervised Learning**:
   - Leverage unlabeled data to improve representation in minority classes.
2. **Ensemble Models**:
   - Combine BERT and GPT-2 to leverage their complementary strengths.
3. **Automated Cleaning Pipelines**:
   - Develop dynamic preprocessing techniques that retain sentiment-rich context while reducing noise.

---

## **References**

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Radford, A., et al. (2019). Language Models are Few-Shot Learners.
3. Hugging Face Transformers Library: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
4. Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
