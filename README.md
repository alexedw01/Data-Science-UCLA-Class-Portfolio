# Overview

This project utilizes the Heart Failure Clinical Records dataset from Kaggle to investigate factors associated with mortality among patients experiencing heart failure. The dataset contains 299 patient records with 12 clinical and demographic features, including age, anaemia status, creatinine phosphokinase levels, diabetes status, ejection fraction, blood pressure, platelet count, serum creatinine, serum sodium, sex, smoking status, and follow-up time. The target variable, death event, indicates whether the patient died during the follow-up period. These features provide a compact yet clinically meaningful basis for analyzing patient risk profiles and predicting adverse outcomes.

Heart failure represents one of the most severe consequences of cardiovascular diseases (CVDs)—the leading cause of death globally, responsible for approximately 17.9 million deaths annually, or 31% of all deaths worldwide. Although many CVD-related outcomes are preventable through lifestyle modification and early risk-factor management, individuals with hypertension, diabetes, hyperlipidaemia, or established cardiovascular conditions remain at elevated risk. Early identification of high-risk patients is therefore critical, and machine learning models offer a scalable, data-driven approach to support clinical decision making.

In this context, our team aims to develop a predictive model for mortality due to heart failure, leveraging the dataset’s clinical attributes to identify patterns associated with increased risk. The objective is to build a reliable and interpretable model that can assist clinicians in early detection, improve patient stratification, and ultimately support better health outcomes.

# Key Methodolgy

After evaluating multiple modeling approaches—including Linear Regression, Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, Principal Component Analysis (PCA), clustering methods, and a feedforward Neural Network—we determined that Random Forest and Logistic Regression were the most effective methodologies for predicting mortality in the heart failure dataset. These models were selected based on their predictive stability, interpretability, and ability to generalize well on a small clinical dataset.

Logistic Regression served as a strong baseline due to its simplicity, interpretability, and compatibility with binary outcomes. Random Forest, an ensemble method that aggregates many decision trees, was chosen as the key predictive model because it consistently handled nonlinear relationships, interacted well with the dataset’s moderately imbalanced structure, and captured feature interactions that simpler models could not. Unlike KNN, which was sensitive to local density and scaling, or Neural Networks, which struggled due to limited data volume, Random Forest demonstrated robustness without requiring extensive hyperparameter tuning or large sample sizes.

# Results, Cross-Validation, Evaluation Metrics, and Conclusions

Each supervised learning model was evaluated using 5-fold cross-validation to measure stability and generalization performance. Evaluation metrics included accuracy, error rate, confusion matrices, true-positive and true-negative rates, F1 scores, ROC curves, and AUC values.

**Linear / Polynomial Regression** produced a not very strong model as expected given linear regression is not expected to do well with binary response variables. 

====== Linear Regression ======
Best degree: 1
Best score (lower MSE = better): 0.1399240332001111
Final Linear Model MSE: 0.12712046426668447
Final Linear Model R2: 0.41683617477905077

====== Ridge Regression ======
Best degree: 1
Best alpha: 10
Best score (lower MSE = better): 0.13969116487842598
Final Ridge Model MSE: 0.1271913792957982
Final Ridge Model R2: 0.41651085281077294

**Logistic Regression** produced consistently strong AUC scores and balanced sensitivity and specificity across folds. Its ROC curves showed reliable discriminative performance, and the model maintained stable accuracy without signs of overfitting. These results, combined with interpretability, made it an effective clinical baseline.

confusion matrix:
array([[40,  2],
       [ 5, 13]])

Accuracy: 0.8833333333333333
Error: 0.1166666666666667
True Positive Rate: 0.7222222222222222
True Negative Rate: 0.9523809523809523
Best Threshold: 0.39076337045217713
AUC: 0.903
average AUC: 0.769
average accuracy: 0.819

**Random Forest** outperformed all other methods. It delivered the highest validation accuracy, strong F1 scores, and the most consistent AUC values across folds. Cross-validation demonstrated minimal variance in performance, indicating strong generalization. The ensemble structure allowed the model to exploit nonlinearities and feature interactions inherent in clinical data. The primary limitations relate to interpretability—Random Forest models are less transparent than logistic models—and to the modest dataset size, which constrains how deeply the trees can grow before risking overfitting. Nonetheless, performance remained stable due to built-in regularization such as bootstrapping and feature subsampling.

Confusion Matrix:
[[161   0]
 [  0  77]]
Prediction Accuracy: 1.0000
Prediction Error: 0.0000
True Positive Rate: 1.0000
True Negative Rate: 1.0000
F1 Score: 1.0000
AUC: 0.9081
Mean AUC: 0.905
Mean Accuracy: 0.8353

Other methods performed less effectively. 

**KNN** achieved moderate accuracy but was affected by scaling and neighborhood sensitivity. 

Best k value: 23
Confusion Matrix:
[[159   2]
 [ 59  18]]
Prediction Accuracy: 0.7437
Prediction Error: 0.2563
True Positive Rate: 0.2338
True Negative Rate: 0.9876
F1 Score: 0.3711
AUC: 0.8889
Mean AUC: 0.8464
Mean Accuracy: 0.7316

**PCA** revealed no dominant directions of variance that could simplify the problem, and **clustering** (with and without PCA transformed data) produced reasonable subgroups but did not align strongly with mortality outcomes. 


Clustering on data without PCA applied:
K-Means Inertia:  3207.2
Total Within-Cluster Sum of Squares for K-Means:  3207.1590519985666
Total Within-Cluster Sum of Squares for Hierarchical Clustering:  3264.6810131913385
Silhouette Score for K-Means:  0.11875064913468546
Silhouette Score for Hierarchical Clustering:  0.106937165682702
Rand Score for K-Means vs Hierarchical Clustering:  0.7522201884617992
Adjusted Rand Score for K-Means vs Hierarchical Clustering:  0.504467227619107

Clustering on data with PCA applied:
K-Means Inertia:  2667.0
Total Within-Cluster Sum of Squares for K-Means:  2667.0265014953716
Total Within-Cluster Sum of Squares for Hierarchical Clustering:  2759.966919425611
Silhouette Score for K-Means:  0.11787052062957105
Silhouette Score for Hierarchical Clustering:  0.08834700200524696
Rand Score for K-Means vs Hierarchical Clustering:  0.7197930083836124
Adjusted Rand Score for K-Means vs Hierarchical Clustering:  0.43958185210978806


**Neural Networks** underperformed due to limited data, class imbalance, and insufficient complexity in the feature space.

--- Fold 1 ---
Epoch 1 | Train Loss: 0.8167, Val Loss: 1.3892, Val Acc: 0.6170
Epoch 5 | Train Loss: 0.8263, Val Loss: 1.4540, Val Acc: 0.5106
Epoch 10 | Train Loss: 0.8178, Val Loss: 1.4571, Val Acc: 0.5106
Epoch 15 | Train Loss: 0.7699, Val Loss: 1.4548, Val Acc: 0.5106
Early stopping at epoch 16
Test Accuracy: 0.5833333333333334

--- Fold 2 ---
Epoch 1 | Train Loss: 0.8087, Val Loss: 1.4390, Val Acc: 0.4468
Epoch 5 | Train Loss: 0.7962, Val Loss: 1.4707, Val Acc: 0.4043
Epoch 10 | Train Loss: 0.7731, Val Loss: 1.4915, Val Acc: 0.2979
Epoch 15 | Train Loss: 0.7646, Val Loss: 1.5056, Val Acc: 0.2766
Early stopping at epoch 16
Test Accuracy: 0.6833333333333333

--- Fold 3 ---
Epoch 1 | Train Loss: 0.8138, Val Loss: 1.8581, Val Acc: 0.1277
Epoch 5 | Train Loss: 0.8268, Val Loss: 2.0919, Val Acc: 0.1277
Epoch 10 | Train Loss: 0.8063, Val Loss: 2.0898, Val Acc: 0.1277
Epoch 15 | Train Loss: 0.7918, Val Loss: 2.0531, Val Acc: 0.1277
Early stopping at epoch 16
Test Accuracy: 0.6833333333333333

--- Fold 4 ---
Epoch 1 | Train Loss: 0.8687, Val Loss: 1.1790, Val Acc: 0.8085
Epoch 5 | Train Loss: 0.8383, Val Loss: 1.0267, Val Acc: 0.8298
Epoch 10 | Train Loss: 0.8044, Val Loss: 1.0321, Val Acc: 0.8511
Epoch 15 | Train Loss: 0.7737, Val Loss: 1.0448, Val Acc: 0.8085
Epoch 20 | Train Loss: 0.7968, Val Loss: 1.0496, Val Acc: 0.7872
Early stopping at epoch 23
Test Accuracy: 0.559322033898305

--- Fold 5 ---
Epoch 1 | Train Loss: 0.8504, Val Loss: 1.4927, Val Acc: 0.4255
Epoch 5 | Train Loss: 0.7725, Val Loss: 1.5052, Val Acc: 0.3617
Epoch 10 | Train Loss: 0.7742, Val Loss: 1.5190, Val Acc: 0.3617
Epoch 15 | Train Loss: 0.7543, Val Loss: 1.5293, Val Acc: 0.3404
Early stopping at epoch 17
Test Accuracy: 0.6440677966101694

# How to Use the Project Code

All code developed for this project is organized to provide a clear and repeatable workflow for working with the heart failure dataset. Users can run the analysis from top to bottom. 

After running the lines for loading the dataset, the user may have to wait a little bit as it is fetching the data from kaggle directly which may take a second or two. 

From there the best advise is to run the code top to bottom as some sections borrow from previous ones. 

Running cells out of top to bottom order may result in different output. 