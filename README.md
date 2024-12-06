Sentiment Analysis of Rotten Tomatoes Movie Reviews
Abstract
This project focuses on sentiment analysis of the Rotten Tomatoes movie review dataset. The objective is to classify movie review phrases into five sentiment categories (Negative, Somewhat Negative, Neutral, Somewhat Positive, Positive). We explored various data preprocessing techniques and evaluated the performance of BERT and GPT-2 models. The experimental results highlight the trade-offs between data cleaning and model performance, with GPT-2 achieving the best results on the original dataset.

Overview
Problem Definition
The goal of this project is to classify movie reviews into discrete sentiment categories to better understand audience feedback. This task is essential for enhancing user experience, especially in recommendation systems and audience sentiment analysis.

Motivation
Understanding the sentiment behind movie reviews is critical for domains like recommendation systems, marketing, and social media monitoring. This task has broader societal implications, as sentiment analysis techniques can also be applied to customer service, political opinion analysis, and mental health monitoring.

Interesting Aspects
This problem is intriguing because it requires parsing human emotions through textual data, which is inherently ambiguous and context-dependent. This research contributes to developing more "human-like" AI systems capable of understanding sentiment and tone in textual communication.

Proposed Approaches
We implemented and compared the following two methods:

BERT: A pre-trained Transformer-based model fine-tuned for text classification.
GPT-2: A generative Transformer model adapted for classification tasks.
Rationale
BERT and GPT-2 are state-of-the-art NLP models that leverage pre-trained knowledge and attention mechanisms to capture contextual dependencies. Our work builds upon prior research, extending it to analyze how data cleaning impacts model performance.

Key Components
Preprocessing: Removing noise (e.g., punctuation, stopwords) and redundant neutral samples.
Models: Using BERT and GPT-2 for classification tasks.
Results: Analyzing the trade-offs between cleaned and original datasets.
Limitations: Over-cleaning caused loss of contextual information, particularly in the Neutral category.
Experiment Setup
Dataset
Source: Rotten Tomatoes movie reviews dataset.
Size: ~156,000 phrases in the original dataset.
Statistics:
Sentiment categories: Negative (10%), Somewhat Negative (20%), Neutral (51%), Somewhat Positive (15%), Positive (4%).
Implementation Details
Models:
BERT: bert-base-uncased.
GPT-2: Adjusted for classification tasks with added pad_token_id.
Hyperparameters:
Learning Rate: 
2
×
1
0
−
5
2×10 
−5
 
Batch Size: 16
Epochs: 3
Environment:
GPU: NVIDIA Tesla V100.
Libraries: PyTorch, Transformers, Scikit-learn.
Model Architecture
BERT: Fine-tuned for classification with tokenized inputs (input_ids and attention_mask).
GPT-2: Configured with pad_token_id for handling padding during batch processing.
Experiment Results
Main Results
Model	Dataset	Accuracy	Weighted F1-Score
BERT	Cleaned	63.78%	64.00%
GPT-2	Cleaned	70.36%	70.00%
GPT-2	Original	69.68%	69.51%
BERT	Original	70.03%	70.00%
Discussion
Analysis
Impact of Data Cleaning:
Cleaning improved precision for Somewhat Negative and Somewhat Positive categories by reducing noise.
However, it caused a loss of contextual information in the Neutral category, impacting performance.
Model Comparison:
GPT-2 on the original dataset achieved consistent performance across epochs with an accuracy of 69.68%.
BERT showed comparable results but struggled slightly with ambiguity in cleaned data.
Limitations
Loss of contextual information for Neutral samples after cleaning.
Limited sample sizes for Positive and Negative categories affected model robustness.
Future Directions
Retain a subset of Neutral samples during cleaning to balance context and noise.
Experiment with ensemble models combining BERT and GPT-2.
Explore semi-supervised techniques to leverage unlabeled test data.
Conclusion
This project demonstrates the effectiveness of Transformer models for sentiment analysis tasks. While GPT-2 achieved strong results on the original dataset (69.68% accuracy, 69.51% F1-Score), the impact of data cleaning highlights the need to balance noise reduction with context preservation. Future work should focus on optimizing data preprocessing and exploring ensemble methods to improve model robustness.

References
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
Radford, A., et al. (2019). Language Models are Few-Shot Learners.
Hugging Face Transformers Library: https://huggingface.co/transformers/
Scikit-learn Documentation: https://scikit-learn.org/
