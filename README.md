
# **Sentiment Analysis Experiment Code**

## **Overview**
This repository contains the code to replicate experiments for the sentiment analysis of Rotten Tomatoes movie reviews. The project uses **BERT** and **GPT-2** models to classify phrases into five sentiment categories:
- Negative
- Somewhat Negative
- Neutral
- Somewhat Positive
- Positive

Both cleaned and uncleaned datasets can be used for training and evaluation. However, **it is recommended to test these datasets separately to avoid potential conflicts**.

---

## **Requirements**
- Python 3.8 or higher
- GPU with CUDA support (optional but recommended for training large models)
- Libraries:
  - Transformers
  - Scikit-learn
  - Pandas
  - PyTorch
  - NLTK

---

## **Best Performing Model**
The file **`ds_project_weight.ipynb`** implements a BERT model with **class-weighted optimization**, addressing the class imbalance issue in the dataset. This model achieved the best performance metrics in our experiments, with the following highlights:
- **Accuracy**: Highest among all tested models.
- **Improved Minority Class Performance**: The weighting mechanism improved results for Positive and Negative categories, which were underrepresented in the dataset.

You can explore the implementation and evaluation results in this notebook for a detailed understanding of how weighting enhances performance.

---

## **Recommendations**
1. **Separate Datasets**: Cleaned and uncleaned datasets should be tested independently to avoid conflicts during data loading.
2. **Hyperparameters**: Adjust learning rate and batch size based on available computational resources.
3. **Hardware**: Use a GPU for faster training, especially for GPT-2.

---

## **References**
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Radford, A., et al. (2019). Language Models are Few-Shot Learners.
- Hugging Face Transformers Library: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)

