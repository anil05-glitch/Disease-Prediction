# Disease Prediction from Symptoms

This project explores the use of machine learning algorithms to predict diseases from symptoms and AI-Powered Explanations . 

### Algorithms Explored

The following algorithms have been explored in code:

1. Naive Bayes
2. Decision Tree
3. Random Forest
4. Gradient Boosting

# Dataset

### Source-1

The dataset for this problem used with the `main.py` script is downloaded from here:

```
https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning
```

This dataset has 133 total columns, 132 of them being symptoms experienced by patiend and last column in prognosis for the same.

### Source-2
The dataset for this problem used with the Jupyter notebook is downloaded from here: 
```
https://impact.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html
```

This dataset has 3 columns:
```
Disease  | Count of Disease Occurrence | Symptom
```

You can either copy paste the whole table from here to an excel sheet or scrape it.

# Directory Structure

```
|_ dataset/
         |_ training_data.csv
         |_ test_data.csv

|_ saved_model/
         |_ [ pre-trained models ]

|_ main.py [ code for laoding kaggle dataset, training & saving the model]

|_ notebook/
         |_ dataset/
                  |_ raw_data.xlsx [Columbia dataset for notebook]
         |_ Disease-Prediction-from-Symptoms-checkpoint.ipynb [ IPython Notebook for loading Columbia dataset, training model and Inference ]
```
## Key Features & Architecture
Interactive Symptom Checker:
Users are presented with a dynamic, organized interface listing an extensive range of symptoms (such as itching, skin rash, joint pain, fatigue, and more). Users can select multiple symptoms checkbox-style to map out their current physical state.

Machine Learning Prediction Engine:
Powered by a trained scikit-learn model (decision_tree.joblib) backed by a symptom-disease training dataset, the app instantly processes selected inputs to output a predicted condition (e.g., Osteoarthritis, Acne, Allergy).

AI-Powered Explanations & Next Steps:
Integrated with Google GenAI, the application is built to provide empathetic, contextual health explanations and actionable advice (such as resting, hydrating, and consulting a certified medical professional).

Responsive UI & Dark Mode:
The interface features a clean, Dark Mode toggle, quick disclaimer/privacy side notes, and fully responsive layouts optimized for both desktop browsers and mobile devices.

Reporting & Export Tools:
Users can seamlessly print or save their analysis results and medical summaries as PDF reports for convenient record-keeping or doctor visits.

**NOTE:** ***This project is for demo purposes only. For any symptoms/disease, please refer to a Doctor.***
