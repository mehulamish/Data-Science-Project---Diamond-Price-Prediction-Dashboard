library(shiny)
library(gbm)
library(ggplot2)
library(plotly)
library(reshape2)
library(data.table)
library(dplyr)
library(caret)

# Load the data
train <- fread("train.csv")
test <- fread("test.csv")

# Convert 'cut', 'color', and 'clarity' variables to factors
train$cut <- as.factor(train$cut)
train$color <- as.factor(train$color)
train$clarity <- as.factor(train$clarity)

test$cut <- as.factor(test$cut)
test$color <- as.factor(test$color)
test$clarity <- as.factor(test$clarity)

train = train %>% 
    mutate(
        volume = x*y*z,
        density = carat / (volume + 0.000001),
        depth_per_volume = depth / (volume + 0.000001),
        depth_per_density = depth / (density + 0.000001),
        depth_per_table = depth / (table + 0.000001)
    ) %>%
    relocate(price, .after = last_col())

test = test %>% 
    mutate(
        volume = x*y*z,
        density = carat / (volume + 0.000001),
        depth_per_volume = depth / (volume + 0.000001),
        depth_per_density = depth / (density + 0.000001),
        depth_per_table = depth / (table + 0.000001)
    )

set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(train$price, p = 0.8, list = FALSE)
train80 <- train[trainIndex, ]
test20 <- train[-trainIndex, ]
print(dim(train))
print(dim(train80))
print(dim(test20))

# UI
ui <- fluidPage(
  titlePanel("Gemstone Price Dataset Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h4("Choose a dataset:"),
      selectInput("dataset", "Dataset",
                  choices = c("Train", "Test")),
      br(),
      hr(),
      h4("Choose a plot type:"),
      selectInput("plot_type", "Plot Type",
                  choices = c("Cut Value Counts (Bar)", "Color Value Counts (Bar)", "Clarity Value Counts (Bar)",
                              "Cut vs Price (Boxplot)", "Color vs Price (Boxplot)", "Clarity vs Price (Boxplot)")),
      br(),
      hr(),
      h5("Root Mean Squared Error"),
      verbatimTextOutput("rmse"),
      h5("Missing Values"),
      verbatimTextOutput("missing_values"),
      h5("Value Counts"),
      verbatimTextOutput("value_counts"),
      br(),
      hr(),
      h4("Choose corr column"),
      selectInput("corr_column", "Correlation column",
                  choices = c("x", "y", "z", "color", "clarity", "carat", "cut", "depth", "depth_per_volume", "depth_per_density", "depth_per_table")),
      width = 3
    ),
    mainPanel(
      plotlyOutput("plot"),
      br(),
      hr(),
      plotlyOutput("corrplot", height="800px"),
      br(),
      hr(),
      plotlyOutput("comparisonPlot", height="1000px"),
      br(),
      hr(),
      plotlyOutput("priceCompPlot", height="1000px"),  # Add this line
      width = 9
    )
  )
)


# Server
server <- function(input, output, session) {
  dataset <- reactive({
    if (input$dataset == "Train") {
      return(train)
    } else {
      return(test)
    }
  })

  preds <- reactiveVal()

  #Reactive function to calculate RMSE
  rmse <- reactive({
    if (is.null(dataset())) return(NULL)
    
    if (input$dataset == "Train" && "price" %in% colnames(dataset())) {
      # Train Gradient Boosting Regressor model
      model <- gbm(price ~ ., data = train80, n.trees = 100, shrinkage = 0.08, interaction.depth = 5, distribution = "gaussian")
      
      # Make predictions on the test set
      preds(predict(model, newdata = test20, n.trees = 100))
      
      # Calculate RMSE
      rmse_val <- sqrt(mean((preds() - test20$price)^2))
      rmse_val
    } else {
      print("No price column detected")
      return(0)
    }
})
  
  # Display RMSE
  output$rmse <- renderPrint({
    rmse_val <- rmse()
    if (!is.null(rmse_val)) {
      paste("Root Mean Squared Error:", round(rmse_val, 2))
    }
  })
  
  # Check for missing values
  output$missing_values <- renderPrint({
    if (!is.null(dataset())) {
      missing_vals <- colSums(is.na(dataset()))
      missing_vals
    }
  })
  
  # Visualize count of different categories
  output$value_counts <- renderPrint({
    if (!is.null(dataset())) {
      value_counts <- table(dataset()$cut)
      value_counts
    }
  })

  observe({
  if (input$dataset == "Train") {
    updateSelectInput(session, "plot_type", choices = c("Cut Value Counts (Bar)", "Color Value Counts (Bar)", "Clarity Value Counts (Bar)",
                                                        "Cut vs Price (Boxplot)", "Color vs Price (Boxplot)", "Clarity vs Price (Boxplot)"))
  } else {
    updateSelectInput(session, "plot_type", choices = c("Cut Value Counts (Bar)", "Color Value Counts (Bar)", "Clarity Value Counts (Bar)"))
  }
  })

# Function to generate bar plots
  generate_bar_plot <- function(var, title) {
    if (var %in% colnames(dataset())) {
      gg <- ggplot(dataset(), aes_string(x = var, fill = var)) +
        geom_bar() +
        labs(title = title) +
        theme_minimal() +
        theme(legend.position="none")
    } else {
      gg <- ggplot() + theme_void()
    }
    return(gg)
  }

  # Function to generate box plots
  generate_box_plot <- function(var, title) {
    if (input$dataset == "Train" && var %in% colnames(dataset())) {
      gg <- ggplot(dataset(), aes_string(x = var, y = "price", fill = var)) +
        geom_boxplot() +
        labs(title = title) +
        theme_minimal() +
        theme(legend.position="none")
    } else {
      gg <- ggplot() + theme_void()
    }
    return(gg)
  }
  # Plotting function based on selected plot type
  output$plot <- renderPlotly({
    if (is.null(dataset())) return(NULL)
    
    plot <- switch(input$plot_type,
                  "Cut Value Counts (Bar)" = generate_bar_plot("cut", "Cut Value Counts (Bar)"),
                  "Color Value Counts (Bar)" = generate_bar_plot("color", "Color Value Counts (Bar)"),
                  "Clarity Value Counts (Bar)" = generate_bar_plot("clarity", "Clarity Value Counts (Bar)"),
                  "Cut vs Price (Boxplot)" = generate_box_plot("cut", "Cut vs Price (Boxplot)"),
                  "Color vs Price (Boxplot)" = generate_box_plot("color", "Color vs Price (Boxplot)"),
                  "Clarity vs Price (Boxplot)" = generate_box_plot("clarity", "Clarity vs Price (Boxplot)")
    ) 
    ggplotly(plot)
  })

  output$corrplot <- renderPlotly({
    if (is.null(dataset())) return(NULL)
    plottedGraph <- dataset() %>% 
      select(-c(cut, color, clarity))
    
    correlation_matrix <- cor(plottedGraph, use = "pairwise.complete.obs")
    melted_cormat <- melt(correlation_matrix)
    
    p <- ggplot(data = melted_cormat, aes(x = Var1, y = Var2, fill = value, label = round(value, 2))) +
      geom_tile(color = "white") +
      geom_text(aes(label = ifelse(value != 0, as.character(round(value, 2)), "")), color = "black", size = 3) +  # Display values in boxes
      scale_fill_gradient2(low = "steelblue", high = "darkred", mid = "white", 
                           midpoint = 0, limit = c(-1, 1), space = "Lab", 
                           name = "Pearson\nCorrelation") +
      theme_minimal() + 
      theme(
        axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1, color = "darkblue"),
        axis.text.y = element_text(size = 12, color = "darkblue"),
        panel.grid.major = element_line(color = "grey", linetype = "dashed"),
        panel.background = element_rect(fill = "aliceblue"),
        plot.title = element_text(hjust = 0.5, size = 20)
      ) +
      labs(title = "Correlation Heatmap", x = "Variable 1", y = "Variable 2")
    
    # Convert to a plotly object for interactivity
    ggplotly(p)
  })

  output$comparisonPlot <- renderPlotly({
    req(input$corr_column)
    if (is.null(dataset()) || !(input$corr_column %in% colnames(dataset())) || !("price" %in% colnames(dataset()))) return(NULL)
    
    p <- ggplot(dataset(), aes_string(x = input$corr_column, y = "price")) +
      geom_point() +
      labs(title = paste("Comparison of", input$corr_column, "and Price"),
           x = input$corr_column,
           y = "Price") +
      theme_minimal()
    
    ggplotly(p)
  })

  output$priceCompPlot <- renderPlotly({
  req(preds())
  p <- ggplot() +
    geom_point(aes(x = test20$price, y = preds()), size = 1, color = "blue") +  # make dots smaller and blue
    geom_abline(intercept = 0, slope = 1, color = "red") +  # add a red diagonal line
    labs(title = "Actual vs. Predicted Price",
         x = "Actual Price",
         y = "Predicted Price") +
    theme_minimal()
  
  ggplotly(p)
})

}

shinyApp(ui, server)