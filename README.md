# Project 4: Titanic Tidymodels Shiny App

## Run the application

Open this folder in RStudio and run:

```r
shiny::runApp()
```

Click **Fit all models** in the Models tab before viewing comparisons or downloading predictions.

## Render the report

```r
quarto::quarto_render("report.qmd")
```

The report uses `embed-resources: true`, so the HTML output can be submitted as a self-contained file.

## Required packages

```r
install.packages(c("shiny", "shinydashboard", "tidyverse", "tidymodels",
                   "janitor", "DT", "ranger", "C50", "kknn", "klaR"))
```

The `Provided_Tidymodels_Code` folder contains the supplied Midterm Tidymodels examples used as the starting point for this project.
