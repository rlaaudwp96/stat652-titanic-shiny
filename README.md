# Titanic Survival Prediction

An interactive **R/Shiny** classification app that compares five models for Titanic survival prediction: logistic regression, k-nearest neighbors, decision tree, random forest, and naive Bayes.

## Run the Shiny app

Open the repository in RStudio and run:

```r
shiny::runApp()
```

The workflow uses an 80/20 stratified train-validation split, median/mode imputation, one-hot encoding, and accuracy, precision, recall, F1, and ROC AUC for evaluation. Click **Fit all models** in the Models tab before viewing comparisons or downloading predictions.

## Required R packages

```r
install.packages(c("shiny", "shinydashboard", "tidyverse", "tidymodels",
                   "janitor", "DT", "ranger", "C50", "kknn", "klaR"))
```

## Repository contents

- `app.R` - Primary R/Shiny application
- `train.csv`, `test.csv` - Titanic training and test data
- `app.py` - Supplemental Streamlit conversion
- `titanic_analysis_python.py` - Supplemental Python conversion
- `requirements.txt` - Python dependencies for the supplemental workflow
