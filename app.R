library(shiny)
library(shinydashboard)
library(tidyverse)
library(tidymodels)
library(discrim)
library(janitor)
library(DT)
library(ranger)
library(C50)
library(kknn)
library(klaR)

set.seed(652)

# -----------------------------
# 1. Read and prepare the data
# -----------------------------
train_raw <- read_csv("train.csv", show_col_types = FALSE) |> clean_names()
test_raw  <- read_csv("test.csv", show_col_types = FALSE) |> clean_names()

train_data <- train_raw |>
  dplyr::select(-any_of(c("name", "ticket", "cabin"))) |>
  mutate(
    survived = factor(survived, levels = c(0, 1)),
    pclass = factor(pclass),
    sex = factor(sex),
    embarked = factor(embarked)
  )

test_data <- test_raw |>
  dplyr::select(-any_of(c("name", "ticket", "cabin"))) |>
  mutate(
    pclass = factor(pclass, levels = levels(train_data$pclass)),
    sex = factor(sex, levels = levels(train_data$sex)),
    embarked = factor(embarked, levels = levels(train_data$embarked))
  )

split <- initial_split(train_data, prop = 0.80, strata = survived)
training_data <- training(split)
validation_data <- testing(split)

base_recipe <- recipe(survived ~ ., data = training_data) |>
  step_rm(any_of("passenger_id")) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors())

make_model <- function(name) {
  switch(
    name,
    "Logistic Regression" = logistic_reg() |> set_engine("glm") |> set_mode("classification"),
    "k-Nearest Neighbors" = nearest_neighbor(neighbors = 7) |> set_engine("kknn") |> set_mode("classification"),
    "Decision Tree" = decision_tree(tree_depth = 5, min_n = 10) |> set_engine("C5.0") |> set_mode("classification"),
    "Random Forest" = rand_forest(trees = 300, min_n = 5) |> set_engine("ranger", importance = "impurity") |> set_mode("classification"),
    "Naive Bayes" = naive_Bayes() |> set_engine("klaR") |> set_mode("classification")
  )
}

make_workflow <- function(name) {
  rec <- base_recipe
  if (name == "Logistic Regression") {
    rec <- rec |> step_dummy(all_nominal_predictors()) |> step_normalize(all_numeric_predictors())
  }
  if (name == "k-Nearest Neighbors") {
    rec <- rec |> step_dummy(all_nominal_predictors()) |> step_normalize(all_numeric_predictors())
  }
  workflow() |> add_recipe(rec) |> add_model(make_model(name))
}

model_names <- c("Logistic Regression", "k-Nearest Neighbors", "Decision Tree", "Random Forest", "Naive Bayes")

fit_all_models <- function() {
  map(set_names(model_names), function(name) {
    tryCatch(
      make_workflow(name) |> fit(data = training_data),
      error = function(e) structure(list(message = conditionMessage(e)), class = "model_error")
    )
  })
}

predict_one_model <- function(fit, new_data) {
  if (inherits(fit, "model_error")) return(NULL)
  class_pred <- predict(fit, new_data, type = "class")
  prob_pred <- predict(fit, new_data, type = "prob")
  bind_cols(new_data |> dplyr::select(survived), class_pred, prob_pred)
}

metric_table <- function(results) {
  map_dfr(names(results), function(name) {
    p <- results[[name]]
    if (is.null(p)) return(tibble(Model = name, Accuracy = NA, Precision = NA, Recall = NA, F1 = NA, ROC_AUC = NA))
    tibble(
      Model = name,
      Accuracy = accuracy(p, survived, .pred_class) |> pull(.estimate),
      Precision = precision(p, survived, .pred_class, event_level = "second") |> pull(.estimate),
      Recall = recall(p, survived, .pred_class, event_level = "second") |> pull(.estimate),
      F1 = f_meas(p, survived, .pred_class, event_level = "second") |> pull(.estimate),
      ROC_AUC = roc_auc(p, survived, .pred_1, event_level = "second") |> pull(.estimate)
    )
  }) |> mutate(across(where(is.numeric), ~ round(.x, 4))) |> arrange(desc(ROC_AUC))
}

ui <- dashboardPage(
  dashboardHeader(title = "Titanic Survival Classifier"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("home")),
      menuItem("Data", tabName = "data", icon = icon("table")),
      menuItem("Models", tabName = "models", icon = icon("cogs")),
      menuItem("Comparison", tabName = "comparison", icon = icon("chart-line")),
      menuItem("Predict", tabName = "predict", icon = icon("ship"))
    )
  ),
  dashboardBody(
    tags$head(tags$style(HTML(".content-wrapper {background-color:#f7f9fb;} .small-box h3 {font-size:26px;}"))),
    tabItems(
      tabItem("overview",
        h2("Titanic passenger survival prediction"),
        p("This application uses tidymodels recipes and workflows to compare five binary classification models."),
        fluidRow(
          valueBoxOutput("n_train"), valueBoxOutput("n_validation"), valueBoxOutput("best_model")
        ),
        box(title = "Five-step data science process", width = 12, status = "primary", solidHeader = TRUE,
            tags$ol(
              tags$li(strong("Collect"), ": Titanic passenger records."),
              tags$li(strong("Explore and prepare"), ": remove identifiers and impute missing values."),
              tags$li(strong("Train"), ": fit five classification algorithms."),
              tags$li(strong("Evaluate"), ": compare Accuracy, Precision, Recall, F1, and ROC AUC."),
              tags$li(strong("Improve and use"), ": select the best model and predict new passengers.")
            ))
      ),
      tabItem("data",
        h2("Data exploration"),
        fluidRow(box(title = "Training data", width = 8, DTOutput("data_table")),
                 box(title = "Missing values", width = 4, tableOutput("missing_table"))),
        box(title = "Outcome distribution", width = 12, plotOutput("outcome_plot"))
      ),
      tabItem("models",
        h2("Train the models"),
        actionButton("fit_models", "Fit all models", class = "btn-primary", icon = icon("play")),
        br(), br(),
        verbatimTextOutput("training_status"),
        DTOutput("model_metrics")
      ),
      tabItem("comparison",
        h2("Model comparison"),
        DTOutput("comparison_table"),
        fluidRow(box(title = "ROC curves", width = 8, plotOutput("roc_plot", height = "500px")),
                 box(title = "Best model", width = 4, verbatimTextOutput("best_explanation"))),
        box(title = "Confusion matrix for selected model", width = 12,
            selectInput("cm_model", "Model", choices = model_names), plotOutput("cm_plot"))
      ),
      tabItem("predict",
        h2("Make a prediction"),
        fluidRow(
          box(title = "Passenger information", width = 5,
              selectInput("pclass", "Passenger class", choices = c(1, 2, 3), selected = 3),
              selectInput("sex", "Sex", choices = c("female", "male")),
              numericInput("age", "Age", 30, min = 0, max = 100),
              numericInput("sib_sp", "Siblings/spouses aboard", 0, min = 0),
              numericInput("parch", "Parents/children aboard", 0, min = 0),
              numericInput("fare", "Fare", 25, min = 0),
              selectInput("embarked", "Embarked", choices = c("S", "C", "Q")),
              selectInput("prediction_model", "Model", choices = model_names, selected = "Random Forest"),
              actionButton("predict_passenger", "Predict survival", class = "btn-success")),
          box(title = "Prediction", width = 7, uiOutput("passenger_prediction")),
          box(title = "Test-data predictions", width = 12, downloadButton("download_predictions", "Download CSV"), DTOutput("test_predictions"))
        )
      )
    )
  )
)

server <- function(input, output, session) {
  fitted <- reactiveVal(NULL)
  predictions <- reactiveVal(NULL)
  status <- reactiveVal("Models have not been fitted yet. Click 'Fit all models'.")

  output$n_train <- renderValueBox(valueBox(nrow(training_data), "Training observations", icon = icon("database"), color = "aqua"))
  output$n_validation <- renderValueBox(valueBox(nrow(validation_data), "Validation observations", icon = icon("flask"), color = "yellow"))
  output$best_model <- renderValueBox({
    m <- predictions(); label <- if (is.null(m)) "Not fitted" else m |> metric_table() |> slice(1) |> pull(Model)
    valueBox(label, "Best by ROC AUC", icon = icon("trophy"), color = "green")
  })

  output$data_table <- renderDT(datatable(head(train_data, 20), options = list(scrollX = TRUE, pageLength = 10)))
  output$missing_table <- renderTable({
    tibble(variable = names(train_raw), missing = map_int(train_raw, ~ sum(is.na(.x)))) |> filter(missing > 0)
  })
  output$outcome_plot <- renderPlot({
    ggplot(train_data, aes(survived, fill = survived)) + geom_bar() + theme_minimal() + labs(x = "Survived", y = "Count", fill = "Survived")
  })
  output$training_status <- renderText(status())

  observeEvent(input$fit_models, {
    withProgress(message = "Fitting classification models", value = 0, {
      fits <- fit_all_models()
      incProgress(0.7)
      preds <- map(fits, predict_one_model, new_data = validation_data)
      fitted(fits); predictions(preds)
      status("Completed: all models were fitted on the training split and evaluated on the validation split.")
      incProgress(1)
    })
  })

  output$model_metrics <- renderDT({
    req(predictions()); datatable(metric_table(predictions()), rownames = FALSE, options = list(pageLength = 10))
  })
  output$comparison_table <- renderDT({
    req(predictions()); datatable(metric_table(predictions()), rownames = FALSE, options = list(pageLength = 10))
  })
  output$best_explanation <- renderText({
    req(predictions()); tbl <- metric_table(predictions()); b <- tbl |> slice(1)
    paste0(b$Model, " is selected because it has the largest validation ROC AUC (", b$ROC_AUC,
           "). ROC AUC measures how well the model separates survivors from non-survivors across classification thresholds.")
  })
  output$roc_plot <- renderPlot({
    req(predictions())
    d <- map2_dfr(predictions(), names(predictions()), ~ if (is.null(.x)) tibble() else roc_curve(.x, survived, .pred_1) |> mutate(Model = .y))
    ggplot(d, aes(1 - specificity, sensitivity, color = Model)) + geom_path(linewidth = 1) + geom_abline(linetype = 2) + coord_equal() + theme_minimal() + labs(x = "1 - Specificity", y = "Sensitivity")
  })
  output$cm_plot <- renderPlot({
    req(predictions()); p <- predictions()[[input$cm_model]]; req(!is.null(p))
    autoplot(conf_mat(p, survived, .pred_class), type = "heatmap") + theme_minimal()
  })

  passenger <- eventReactive(input$predict_passenger, {
    req(fitted())
    new_data <- tibble(
      passenger_id = 0L,
      pclass = factor(input$pclass, levels = levels(train_data$pclass)),
      sex = factor(input$sex, levels = levels(train_data$sex)),
      age = input$age, sib_sp = input$sib_sp, parch = input$parch, fare = input$fare,
      embarked = factor(input$embarked, levels = levels(train_data$embarked))
    )
    fit <- fitted()[[input$prediction_model]]
    bind_cols(predict(fit, new_data, type = "class"), predict(fit, new_data, type = "prob"))
  })
  output$passenger_prediction <- renderUI({
    req(passenger()); p <- passenger(); survived <- as.character(p$.pred_class) == "1"
    div(style = paste0("padding:30px;text-align:center;border-radius:10px;background:", ifelse(survived, "#d9f2df", "#f8d7da")),
        h2(ifelse(survived, "Predicted: SURVIVED", "Predicted: DID NOT SURVIVE")),
        h4(paste0("Estimated survival probability: ", round(100 * p$.pred_1, 1), "%")))
  })
  output$test_predictions <- renderDT({
    req(fitted()); fit <- fitted()[[input$prediction_model]]
    datatable(bind_cols(test_raw |> dplyr::select(passenger_id), predict(fit, test_data, type = "class"), predict(fit, test_data, type = "prob")), options = list(scrollX = TRUE, pageLength = 10))
  })
  output$download_predictions <- downloadHandler(
    filename = function() paste0("titanic_", gsub(" ", "_", tolower(input$prediction_model)), "_predictions.csv"),
    content = function(file) {
      req(fitted()); fit <- fitted()[[input$prediction_model]]
      out <- bind_cols(test_raw |> dplyr::select(passenger_id), predict(fit, test_data, type = "class"), predict(fit, test_data, type = "prob"))
      write_csv(out, file)
    }
  )
}

shinyApp(ui, server)
