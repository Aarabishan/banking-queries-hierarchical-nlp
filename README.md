# 🏦 Hierarchical Multi-task Neural Network for Banking Customer Service

## 📌 Project Overview

This project focuses on building an intelligent banking query understanding system using a hierarchical multi-task deep learning approach.
The system is designed to automatically:

- Categorize customer queries into major banking domains
- Classify specific customer intents within each domain
- Predict urgency levels to support priority-based handling

All tasks are solved simultaneously using a single hierarchical 3-headed neural network, improving efficiency, consistency, and real-world applicability.

## 🧠 Problem Definition

Banking institutions process millions of customer service inquiries daily, requiring accurate intent 
understanding and assessing urgency for accurate routing, effective resource allocation and timely 
response. Traditional flat classification system may fail to simultaneously capture the both 
customer intent and the service priority, leading to delayed response, misrouting queries, and 
suboptimal customer experience.

## 🎯 Objectives (Proposed Solution)

- Develop a hierarchical multi-task neural network to classify banking queries by
domain, customer intent, and urgency.
- Deploy a functional web application integrating the model with Explainable AI
(XAI) for transparent usage.
- Experiment with a Generative AI model to evaluate advanced performance and
compare with the custom neural network.

## 📈 Business Impact

The proposed solution provides multiple operational advantages such as: 

- Intelligent Routing: Enables routing to specialized teams to automatically. 
- Priority Management: Ensures critical issues receive the immediate attention by urgency 
assessment. 
- Operational Efficiently: Hierarchical processing reduces the decision delay and complexity 
and improves accuracy. 
- Cost optimization: Multi-tasking reduces the infrastructure requirements and maintenance 
cost compared to separate systems.

## 📂 Dataset

- Source: Banking77 dataset
- Size: 77 distinct intents
- Labels:
  - Categories (4):
      - Card Services
      - Account Management
      - Transfer & Payments
      - General Support
  - Urgency Levels (3):

      - Critical
      - Normal
      - Low

## 🔧 Data Preprocessing

- Text cleaning and normalization
-  FeatureEngineering:
    - Categorized queries into domains: Card, Account, Transfer, Support
    - Created urgency labels: Critical, Normal, Low
- Tokenization and padding
- Label encoding for:
    - Categories
    - Intents
    - Urgency levels
- Mapping domain-specific keywords to assist task definition

## 📊 Exploratory Data Analysis (EDA)

## 🏗️ Model Development

  1️⃣ Baseline Model
  - Single-head LSTM
  - Intent classification only
  
  2️⃣ Basic Multi-Task Model
  - Shared embedding + LSTM
  - Three independent output heads:
      - Category
      - Intent
      - Urgency
  
  3️⃣ Advanced Hierarchical Model (Final)
  - Shared embedding + BiLSTM layers
  - Hierarchical flow:
        - Category → Intent → Urgency
  
  - Domain-specific intent branches
  - Global max & average pooling
  - Multi-task loss optimization

## ⚙️ Optimization & Regularization

To address overfitting and generalization issues:
  - Dropout increased across shared and task-specific layers
  - L2 regularization added to dense layers
  - Loss weighting adjusted to emphasize urgency prediction
  - Early stopping and learning rate scheduling applied

## 📈 Model Evaluation

- Accuracy, precision, recall, and F1-score per task
- Loss gap and convergence analysis
- Confusion matrix analysis
- Comparison across:
    -  Baseline
    -  Basic multi-task
    -  Hierarchical optimized model

The final model shows:
    - Improved intent accuracy
    - Reduced overfitting
    - Better hierarchical consistency
    - Strong performance on critical urgency detection

Final Model Scores:

| Head | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) |
|---|---:|---:|---:|---:|
| Category | **94.55%** | **94.55%** | **94.55%** | **94.54%** |
| Intent | **81.40%** | **82.46%** | **81.40%** | **81.29%** |
| Urgency | **75.71%** | **74.64%** | **75.71%** | **74.62%** |

As the proposed architecture is a three-headed multi-task model, performance is evaluated separately for Category, Intent, and Urgency. A single overall accuracy does not fully represent the performance of all three prediction tasks.



## 🔍 Explainability (XAI)

- Integrated explainability techniques into the web application
- Enables interpretation of model predictions
- Helps understand which parts of the query influence:
    - Category
    - Intent
    - Urgency


## 🤖 GenAI Integration

- Experimented with Hugging Face transformer models
- Used for:
    - Performance comparison
    - Exploring zero-shot / few-shot capabilities
- Highlights strengths and limitations compared to task-specific ANN models

🌐 Deployment

- Deployed as a web application
- Features:
    - Real-time query prediction
    - Hierarchical output display
    - Explainability visualization
- Implemented using:
    - Streamlit
    - Ngrok (for Colab-based deployment)

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Hugging Face Transformers
- Google Colab

## 📌 Key Takeaways

- Hierarchical multi-task learning improves consistency and performance
- Intent prediction benefits significantly from category context
- Urgency prediction remains challenging due to class imbalance
- Regularization and loss weighting are critical for generalization
- Task-specific deep learning models outperform generic GenAI for structured tasks

## 🚀 Future Improvements

- Better urgency preprocessing and labeling
- Data augmentation for rare intents
- Advanced transformer-based hierarchical models
- Real-time deployment with scalable backend
- Enhanced explainability techniques


🎓 Acadamice context: This repository documents one of the applied projects completed for the Artificial Intelligence module in my MSc in Data Science programme. All results, code, and design choices are provided for educational purposes; actual performance may vary across environments and dataset revisions.
