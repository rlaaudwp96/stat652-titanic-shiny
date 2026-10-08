# Titanic ML Shiny App - Enhanced Version
# A front end for running machine learning classification models on the Titanic dataset
# Features: Hyperparameter tuning, cross-validation, model comparison

library(shiny)
library(shinydashboard)
library(tidymodels)
library(titanic)
library(tidyverse)
library(janitor)
library(naniar)
library(discrim)
library(yardstick)
library(ggplot2)
library(DT)

data(titanic_train)
data(titanic_test)

titanic_train <- titanic_train |> clean_names()
titanic_test <- titanic_test |> clean_names()

ui <- dashboardPage(
  skin = "green",
  dashboardHeader(title = "Titanic ML Classification"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data", tabName = "data", icon = icon("database")),
      menuItem("Preprocessing", tabName = "preprocess", icon = icon("cogs")),
      menuItem("Model Tuning", tabName = "tuning", icon = icon("sliders")),
      menuItem("Results", tabName = "models", icon = icon("chart-bar")),
      menuItem("Comparison", tabName = "comparison", icon = icon("balance-scale")),
      menuItem("Predict", tabName = "predict", icon = icon("question-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "data",
        fluidRow(box(title = "Training Data", status = "primary", solidHeader = TRUE, width = 12, DTOutput("data_head"))),
        fluidRow(box(title = "Missing Values", status = "info", solidHeader = TRUE, width = 6, plotOutput("vis_miss")),
                 box(title = "Missing %", status = "info", solidHeader = TRUE, width = 6, plotOutput("gg_miss")))
      ),
      tabItem(tabName = "preprocess",
        fluidRow(box(title = "Variables", status = "primary", width = 6,
              checkboxGroupInput("predictors", "Select:",
                choices = c("pclass","sex","age","sib_sp","parch","fare","embarked"),
                selected = c("pclass","sex","age","sib_sp","parch","fare","embarked"))),
          box(title = "Options", status = "primary", width = 6,
              checkboxInput("impute", "Impute age", value = TRUE),
              checkboxInput("normalize", "Normalize", value = TRUE),
              sliderInput("split", "Train/Test:", 0.5, 0.95, 0.8, 0.05))),
        fluidRow(box(title = "Recipe", status = "info", width = 12, verbatimTextOutput("recipe_sum")))
      ),
      tabItem(tabName = "tuning",
        fluidRow(box(title = "Model", status = "primary", width = 4,
              selectInput("model_sel", "Choose:",
                choices = c("Null Model","C5.0","XGBoost","Random Forest","Logistic Regression","Naive Bayes"),
                selected = "Random Forest"),
              uiOutput("hyperpars"),
              checkboxInput("use_cv", "Cross-Validation", value = TRUE),
              actionButton("train_one", "Train Model", class = "btn-primary w-100"),
              br(), br(),
              actionButton("train_all", "Train All", class = "btn-success w-100")),
          box(title = "Status", status = "info", width = 8, verbatimTextOutput("status")))
      ),
      tabItem(tabName = "models",
        fluidRow(box(title = "Select", width = 3, selectInput("res_model", "Model:", choices = c("Null Model","C5.0","XGBoost","Random Forest","Logistic Regression","Naive Bayes"), selected = "Random Forest")),
                 box(title = "Metrics", width = 9, tableOutput("metrics"))),
        fluidRow(box(title = "Confusion Matrix", width = 6, plotOutput("cm")),
                 box(title = "ROC", width = 6, plotOutput("roc"))),
        fluidRow(box(title = "Probabilities", width = 12, plotOutput("dens")))
      ),
      tabItem(tabName = "comparison",
        fluidRow(box(title = "ROC Curves", width = 12, plotOutput("comb_roc", height = "400px"))),
        fluidRow(box(title = "Comparison", width = 12, DTOutput("comp_table")))
      ),
      tabItem(tabName = "predict",
        fluidRow(box(title = "Passenger", width = 6,
              selectInput("p_pclass","Pclass:",c(1,2,3)), selectInput("p_sex","Sex:",c("male","female")),
              numericInput("p_age","Age:",30), numericInput("p_sibsp","SibSp:",0),
              numericInput("p_parch","Parch:",0), numericInput("p_fare","Fare:",50),
              selectInput("p_emb","Embarked:",c("C","Q","S"))),
          box(title = "Predict", width = 6,
              selectInput("p_model","Model:",c("Null Model","C5.0","XGBoost","Random Forest","Logistic Regression","Naive Bayes"), selected = "Random Forest"),
              uiOutput("pred_result"))),
        fluidRow(box(title = "Test Predictions", width = 12, DTOutput("test_preds"), downloadButton("dl_preds","Download")))
      )
    )
  )
)

server <- function(input, output, session) {
  rv <- reactiveValues(results = list(), trained = FALSE)
  
  get_data <- reactive({
    train <- titanic_train |> mutate(survived=as_factor(survived),pclass=as_factor(pclass),sex=as_factor(sex),embarked=as_factor(embarked))
    test <- titanic_test |> mutate(pclass=as_factor(pclass),sex=as_factor(sex),embarked=as_factor(embarked))
    list(train=train,test=test)
  })
  
  split_data <- reactive({ set.seed(123); initial_split(get_data()$train, prop=input$split, strata=survived) })
  
  make_recipe <- function(tr_data, for_glm=FALSE) {
    rec <- recipe(survived~., data=tr_data)
    for(v in setdiff(c("pclass","sex","age","sib_sp","parch","fare","embarked"), input$predictors)) {
      rec <- rec |> step_rm(!!sym(v))
    }
    if(input$impute) rec <- rec |> step_impute_mean(age)
    if(input$normalize) rec <- rec |> step_normalize(all_numeric_predictors())
    if(for_glm) rec <- rec |> step_dummy(all_factor_predictors())
    rec |> prep()
  }
  
  output$hyperpars <- renderUI({
    switch(input$model_sel,
      "C5.0" = tagList(sliderInput("c50_t","Trees:",5,100,20,5)),
      "XGBoost" = tagList(sliderInput("xgb_t","Trees:",5,100,20,5), sliderInput("xgb_lr","LR:",0.01,0.3,0.1,0.01)),
      "Random Forest" = tagList(sliderInput("rf_t","Trees:",50,500,100,10), sliderInput("rf_m","mtry:",1,7,3,1)),
      "Logistic Regression" = tagList(sliderInput("glmp","Penalty:",0.001,1,0.1,0.001), sliderInput("glmm","Mixture:",0,1,0.5,0.1)),
      "Naive Bayes" = tagList(sliderInput("nb_l","Laplace:",0,5,1,0.5), sliderInput("nb_s","Smoothness:",0,2,1,0.1))
    )
  })
  
  output$status <- renderText({ if(rv$trained) paste("Trained:",Sys.time()) else "Press Train to begin" })
  
  train_model <- function(mname, pars=list()) {
    spl <- split_data()
    tr <- training(spl); te <- testing(spl)
    
    # Build recipe based on model type
    # Remove ID columns immediately so they're never used as predictors
    rec <- recipe(survived~., data=tr) |>
      step_rm(passenger_id, name, ticket, cabin)

    for(v in setdiff(c("pclass","sex","age","sib_sp","parch","fare","embarked"), input$predictors)) {
      rec <- rec |> step_rm(!!sym(v))
    }
    if(input$impute) rec <- rec |> step_impute_mean(age)
    
    # Model-specific preprocessing
    if (mname %in% c("XGBoost", "Random Forest")) {
      rec <- rec |> step_normalize(all_numeric_predictors())
    }
    
    if (mname %in% c("Logistic Regression", "C5.0")) {
      rec <- rec |> step_dummy(all_factor_predictors()) |> step_normalize(all_numeric_predictors())
    }
    
    if (mname %in% c("Null Model", "Naive Bayes")) {
      if(input$normalize) rec <- rec |> step_normalize(all_numeric_predictors())
    }
    
    rec <- rec |> prep()
    tr_b <- rec|>bake(tr); te_b <- rec|>bake(te)
    
    # Keep original survived column for metrics
    te_survived <- te$survived
    
    mod <- switch(mname,
      "Null Model" = null_model()|>set_engine("parsnip")|>set_mode("classification"),
      "C5.0" = boost_tree(trees=pars$c50_t%||%20)|>set_engine("C5.0")|>set_mode("classification"),
      "XGBoost" = boost_tree(trees=pars$xgb_t%||%20,learn_rate=pars$xgb_lr%||%0.1)|>set_engine("xgboost")|>set_mode("classification"),
      "Random Forest" = rand_forest(trees=pars$rf_t%||%100,mtry=pars$rf_m%||%3)|>set_engine("ranger")|>set_mode("classification"),
      "Logistic Regression" = logistic_reg(penalty=pars$glmp%||%0.1,mixture=pars$glmm%||%0.5)|>set_engine("glmnet")|>set_mode("classification"),
      "Naive Bayes" = naive_Bayes(Laplace=pars$nb_l%||%1,smoothness=pars$nb_s%||%1)|>set_engine("klaR")|>set_mode("classification")
    )
    fitd <- mod |> fit(survived~., data=tr_b)
    
    # Get predictions - simplified robust handling
    pred_class_vec <- as.character(predict(fitd,te_b)[[1]])
    pred_prob_mat <- predict(fitd,te_b,type="prob")
    
    # Create predictions data frame
    preds <- te_b %>%
      mutate(.pred_class = factor(pred_class_vec, levels = c("0", "1"))) %>%
      bind_cols(data.frame(
        .pred_0 = as.numeric(pred_prob_mat[[1]]),
        .pred_1 = as.numeric(pred_prob_mat[[2]])
      ))
    
    # Add survived column back from original data
    preds$survived <- te_survived
    
    list(model=fitd, preds=preds, recipe=rec)
  }
  
  get_res <- function(mname) {
    r <- rv$results[[mname]]; if(is.null(r)) return(NULL)
    p <- r$preds
    
    # Calculate metrics safely
    met <- p |> metrics(truth=survived,estimate=.pred_class)
    acc_val <- met$.estimate[met$.metric == "accuracy"]
    kap_val <- met$.estimate[met$.metric == "kap"]
    
    # ROC AUC - handle case where it might fail
    auc_val <- tryCatch({
      p |> roc_auc(survived,.pred_0) |> pull(.estimate)
    }, error = function(e) NA)
    
    list(acc=acc_val, kap=kap_val, auc=auc_val,
         cm=conf_mat(p,truth=survived,estimate=.pred_class),
         roc=roc_curve(p,survived,.pred_0), preds=p)
  }
  
  observeEvent(input$train_one, {
    withProgress(message=input$model_sel, {
      pars <- list(c50_t=input$c50_t,xgb_t=input$xgb_t,xgb_lr=input$xgb_lr,
                   rf_t=input$rf_t,rf_m=input$rf_m,glmp=input$glmp,glmm=input$glmm,nb_l=input$nb_l,nb_s=input$nb_s)
      rv$results[[input$model_sel]] <- train_model(input$model_sel, pars)
      rv$trained <- TRUE
    })
  })
  
  observeEvent(input$train_all, {
    mods <- c("Null Model","C5.0","XGBoost","Random Forest","Logistic Regression","Naive Bayes")
    withProgress("Training all...",value=0, {
      for(i in seq_along(mods)) { incProgress(1/length(mods),detail=mods[i])
        rv$results[[mods[i]]] <- train_model(mods[i],list(c50_t=20,xgb_t=20,xgb_lr=0.1,rf_t=100,rf_m=3,glmp=0.1,glmm=0.5,nb_l=1,nb_s=1))
      }
      rv$trained <- TRUE
    })
  })
  
  output$data_head <- renderDT(datatable(head(get_data()$train,20),options=list(scrollX=TRUE)))
  output$vis_miss <- renderPlot(vis_miss(get_data()$train))
  output$gg_miss <- renderPlot(gg_miss_var(get_data()$train,show_pct=TRUE)+theme_minimal())
  output$recipe_sum <- renderPrint({ rec <- make_recipe(training(split_data())); summary(rec) })
  
  output$metrics <- renderTable({ r<-get_res(input$res_model); if(is.null(r)) return(NULL)
    data.frame(Metric=c("Accuracy","Kappa","ROC_AUC"),Value=round(c(r$acc,r$kap,r$auc),4)) },striped=TRUE)
  output$cm <- renderPlot({ r<-get_res(input$res_model); if(is.null(r)) return(NULL)
    autoplot(r$cm,type="heatmap")+scale_fill_gradient(low="white",high="steelblue")+theme_minimal() })
  output$roc <- renderPlot({ r<-get_res(input$res_model); if(is.null(r)) return(NULL)
    autoplot(r$roc)+theme_minimal() })
  output$dens <- renderPlot({ r<-get_res(input$res_model); if(is.null(r)) return(NULL)
    r$preds|>ggplot()+geom_density(aes(x=.pred_0,fill=survived),alpha=0.5)+theme_minimal() })
  
  output$comb_roc <- renderPlot({ if(length(rv$results)==0) return(NULL)
    d <- map2_dfr(rv$results,names(rv$results),~roc_curve(.x$preds,survived,.pred_0)|>mutate(model=.y))
    ggplot(d,aes(x=1-specificity,y=sensitivity,color=model))+geom_path()+geom_abline(lty=2)+coord_equal()+theme_minimal() })
  output$comp_table <- renderDT({ if(length(rv$results)==0) return(NULL)
    d <- map_dfr(names(rv$results),~{r<-get_res(.x);data.frame(Model=.x,Accuracy=round(r$acc,4),Kappa=round(r$kap,4),ROC_AUC=round(r$auc,4))})
    datatable(d,options=list(order=list(list(2,"desc")))) })
  
  # Reactive expression for single passenger prediction
  pred_result_val <- reactive({
    r <- rv$results[[input$p_model]]
    if(is.null(r)) return(NULL)

    tryCatch({
      # Get the terms from the recipe to know what columns are expected
      rec_terms <- r$recipe$term_info
      expected_cols <- rec_terms$variable[rec_terms$role == "predictor"]

      # Build data frame with predictor columns only
      np <- data.frame(pclass=factor(input$p_pclass,1:3))
      if("sex" %in% expected_cols) np$sex <- factor(input$p_sex,c("male","female"))
      if("age" %in% expected_cols) np$age <- input$p_age
      if("sib_sp" %in% expected_cols) np$sib_sp <- input$p_sibsp
      if("parch" %in% expected_cols) np$parch <- input$p_parch
      if("fare" %in% expected_cols) np$fare <- input$p_fare
      if("embarked" %in% expected_cols) np$embarked <- factor(input$p_emb,c("C","Q","S"))

      nd <- r$recipe|>bake(np)

      # Get predictions - handle different model output formats
      pred_class <- predict(r$model, nd)
      pred_prob <- predict(r$model, nd, type = "prob")

      # Extract class prediction (handle both factor and character outputs)
      if(".pred_class" %in% names(pred_class)) {
        pc <- as.character(pred_class$.pred_class[1])
      } else {
        pc <- as.character(pred_class[[1]][1])
      }

      # Extract probability for class "1" (survived)
      if(".pred_1" %in% names(pred_prob)) {
        pp <- as.numeric(pred_prob$.pred_1[1])
      } else if(ncol(pred_prob) >= 2) {
        pp <- as.numeric(pred_prob[[2]][1])
      } else {
        pp <- 0.5
      }

      list(pc = pc, pp = pp, error = NULL)
    }, error = function(e) {
      list(pc = NULL, pp = NULL, error = paste("Error with", input$p_model, ":", conditionMessage(e)))
    })
  })

  output$pred_result <- renderUI({
    val <- pred_result_val()
    if(is.null(val) || !is.null(val$error)) return(NULL)
    col <- if(val$pc=="1") "green" else "red"
    txt <- if(val$pc=="1") "SURVIVED" else "DIED"
    div(style=paste0("background:",col,";color:white;padding:20px;border-radius:10px;text-align:center"),
        h2(txt), h4(paste0(round(val$pp*100,1),"%")))
  })
  
  output$test_preds <- renderDT({ if(length(rv$results)==0) return(NULL)
    r <- rv$results[[input$p_model]]; if(is.null(r)) return(NULL)
    td <- get_data()$test; bt <- r$recipe|>bake(td)
    datatable(predict(r$model,bt)|>bind_cols(td|>select(passenger_id)),options=list(scrollX=TRUE)) })

  output$dl_preds <- downloadHandler("titanic_predictions.csv",function(file){
    if(length(rv$results)==0 || is.null(rv$results[[input$p_model]])) return(NULL)
    r <- rv$results[[input$p_model]]; td <- get_data()$test
    write.csv(predict(r$model,r$recipe|>bake(td))|>bind_cols(td|>select(passenger_id)),file,row.names=FALSE)
  })
}

shinyApp(ui,server)
