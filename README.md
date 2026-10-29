# Titanic Survival Predictor

R/Shiny implementation of a Titanic classification project. The app compares logistic regression, k-nearest neighbors, decision tree, random forest, and naive Bayes models.

## Run the application

Install dependencies and run the app with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For a command-line model comparison, run `python titanic_analysis.py`.

Click **Fit all models** in the Models tab before viewing comparisons or downloading predictions.

## Data and evaluation

The workflow uses an 80/20 stratified train-validation split, median/mode imputation, one-hot encoding, standardized numeric features, and accuracy, precision, recall, F1, and ROC AUC for evaluation.

The primary implementation is R with Shiny for interactive model comparison. A Python conversion is also included as a supplemental reproducible workflow.
