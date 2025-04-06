---
title: "Telco Customer Churn Prediction"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## **Telco Customer Churn Prediction**

### **1. Load Libraries & Set Working Directory**

```{r load-libraries}
library(ROSE)
library(caret)
library(dplyr)
library(ggplot2)
library(rpart.plot)
library(randomForest)

setwd("C:/Users/Tambe/OneDrive/Desktop/data science")
getwd()
```

### **2. Load and Explore Data**

```{r load-data}
f <- read.csv("Telco-Customer-Churn.csv")

# Check dataset structure
str(f)

# Summary statistics
summary(f)
```

### **3. Handle Missing Values**

```{r missing-values}
# Check for missing values
colSums(is.na(f))

# Remove rows with missing values
f <- na.omit(f)
colSums(is.na(f))
```

### **4. Data Visualization**

```{r visualize-data}
# Check class imbalance
prop.table(table(f$Churn))
barplot(prop.table(table(f$Churn)), col=rainbow(2), ylim=c(0,1), main="Churn Distribution")
```

### **5. Feature Engineering**

```{r feature-engineering}
# Categorize tenure
f$tenure_group <- cut(f$tenure, 
                      breaks = c(0, 12, 24, 48, 60, Inf), 
                      labels = c("0-12 Month", "12-24 Month", "24-48 Month", "48-60 Month", "> 60 Month"))

# Convert to factor
to_factor <- c("tenure_group", "Churn")
f[to_factor] <- lapply(f[to_factor], as.factor)

# Drop unnecessary columns
f <- f %>% select(-c(customerID, tenure, TotalCharges))
```

### **6. Handle Class Imbalance Using SMOTE**

```{r smote}
f_balanced <- ROSE::ovun.sample(Churn ~ ., data = f, method = "over")$data

# Check class distribution
prop.table(table(f_balanced$Churn))
barplot(prop.table(table(f_balanced$Churn)), col=rainbow(2), ylim=c(0,1), main="Balanced Churn Data")
```

### **7. Split Data into Training & Testing Sets**

```{r train-test-split}
set.seed(2017)
intrain <- createDataPartition(f_balanced$Churn, p = 0.7, list = FALSE)
training <- f_balanced[intrain,]
testing <- f_balanced[-intrain,]
```

### **8. Train Decision Tree Model**

```{r train-decision-tree}
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

tree_model <- train(Churn ~ ., data = training, method = "rpart", trControl = ctrl, tuneLength = 10)

# Visualize Decision Tree
rpart.plot(tree_model$finalModel, main = "Decision Tree")
```

### **9. Model Evaluation**

```{r evaluate-model}
# Make predictions
predictions <- predict(tree_model, newdata = testing)

# Confusion Matrix
testing$Churn <- factor(testing$Churn, levels = levels(predictions))
confusionMatrix(predictions, testing$Churn)
```

### **10. Entropy Calculation**

```{r entropy-calculation}
calculate_entropy <- function(labels) {
  proportions <- table(labels) / length(labels)
  entropy <- -sum(proportions * log2(proportions), na.rm = TRUE)
  return(entropy)
}

entropy_value <- calculate_entropy(training$Churn)
print(paste("Entropy of Churn:", entropy_value))
```

