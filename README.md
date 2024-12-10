# **Projects Overview: Emotion Analysis and Turkish News Classification**

This repository contains two Natural Language Processing (NLP) projects focusing on text classification, sentiment analysis, and text summarization. These projects showcase advanced preprocessing, model implementation, and result generation using cutting-edge technologies.

---

## **Project 1: Emotion Analysis and Conclusion Generation**

### **Overview**
This project leverages Google's **GoEmotion** dataset to classify text into four primary emotion categories and generate a conclusion for each classified text. The pipeline involves comprehensive preprocessing, training a BERT-based model, and generating summaries using a GPT-2 model.

### **Dataset**
- **Source:** Google’s GoEmotion Dataset  
- **Size:** ~227,000 rows  
- **Emotion Categories:** Reduced from 27 to 4: **Positive**, **Negative**, **Neutral**, and **Mixed**.  

### **Preprocessing Steps**
1. **Duplicate Removal:** Removed repeated records.  
2. **Emotion Reduction:** Consolidated similar categories based on correlation.  
3. **Balancing Classes:** Applied class-based undersampling for balanced training.  
4. **Text Cleaning:** Removed stopwords and contractions, followed by lemmatization.

### **Model Training**
- **Model:** BERT (Base Uncased)  
- **Optimizer:** AdamW  
- **Accuracy Achieved:** ~51%
- 

### **Conclusion Generation**
- **Model Used:** GPT-2  
- **Task:** Generate a conclusion based on the classified emotion.  
- **Pipeline:**  
  1. Classify the text using BERT.  
  2. Generate a conclusion aligned with the predicted emotion.


## **Project 2: Turkish News Classification and Summarization**

### **Overview**
This project deals with a large Turkish news dataset, focusing on classifying news articles into nine categories and generating concise summaries using a specialized summarization model.

### **Dataset**
- **Source:** University-provided Turkish news dataset  
- **Size:** 1.5 million rows  
- **Categories:** 9 news categories  

### **Preprocessing Steps**
1. **Class-Based Filtering:** Filtered rows based on word count thresholds.  
2. **Balancing Classes:** Applied undersampling to equalize the number of rows per class.  
3. **Text Cleaning:**  
   - Removed Turkish stopwords using an extended stopword list.  
   - Performed stemming to reduce words to their root forms.  

### **Model Training**
- **Model:** BERTTürk (BERT model for Turkish)  
- **Tokenizer:** BERTTürk Tokenizer  
- **Optimizer:** AdamW  
- **Accuracy Achieved:** 91%  
- **Validation:** Confusion Matrix
![Screenshot 2024-12-09 184115](https://github.com/user-attachments/assets/5222dbca-9ec6-4dd0-b2f1-e7822f93b8aa)

### **Summarization**
- **Model Used:** Mukayese (Summarization model for Turkish)  
- **Task:** Generate concise summaries of news articles.  
- **Challenges:**  
  - Summarization is often limited to the first few paragraphs.  
  - Adjustments to input token limits improved performance but require further optimization.  



## **Results**
### **Emotion Analysis**
- Emotion Classification Accuracy: ~51%  + GPT-2 Conclusion Genaration
![Screenshot 2024-12-09 122353](https://github.com/user-attachments/assets/62a00d5d-0441-4f58-9f87-b4eb96a1dfaa)


### **Turkish News Classification**
- Emotion Classification Accuracy: ~91%  + Muakyese Summarization

![Screenshot 2024-12-09 180339](https://github.com/user-attachments/assets/98c898e6-a72b-43c9-9e46-3a29aeb0fb29)

