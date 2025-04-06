library(ROSE)
library(caret)

getwd()
setwd("C://Users//Tambe//OneDrive//Desktop//data sceince")
f<-read.csv("Telco-Customer-Churn.csv")

#1.Checking
summary(f)        


any(is.na(f))        
sum(is.na(f))
colSums(is.na(f))        #In which column how many NA is there
sum(is.na(f$TotalCharges))



# Remove missing values
f <- na.omit(f)       #Discard NA values(NA less than 50%)
colSums(is.na(f))#Display the removed NA col to be 0


#2. Imbalanced Data
str(f)    #Structure of data(how many variables nd observation)
(prop.table(table(f$Churn)))  #O/P depend on only churn column hence take prop of churn column only
barplot(prop.table(table(f$Churn)),col=rainbow(2),ylim=c(0,1),main="Churn_Prediction")



# Recode columns
cols_recode1 <- c(10:15)

# Group tenure into categories
group_tenure <- function(tenure){
  if (tenure >= 0 & tenure <= 12){
    return('0-12 Month')
  }else if(tenure > 12 & tenure <= 24){
    
    
    return('12-24 Month')
  }else if (tenure > 24 & tenure <= 48){
    
    
    return('24-48 Month')
  }else if (tenure > 48 & tenure <=60){
    return('48-60 Month')
    
    
  }else if (tenure > 60){
    return('> 60 Month')
  }
}

f$tenure_group <- sapply(f$tenure, group_tenure)
f$tenure_group <- as.factor(f$tenure_group)

# Remove unnecessary columns
f$customerID <- NULL
f$tenure <- NULL
f$TotalCharges <- NULL




# Use SMOTE to balance classes
over <- ROSE::ovun.sample(Churn ~ ., data = f, method = "over")

f <- over$data
levels(f$Churn) <- make.names(levels(f$Churn))
# Check for class imbalance
prop.table(table(f$Churn))       #Proportion of 0 nd 1
barplot(prop.table(table(f$Churn)), col=rainbow(2), ylim=c(0,1), main = "Churn_Prediction")



# Split into training and testing sets
set.seed(2017)
intrain <- createDataPartition(f$Churn, p = 0.7, list = FALSE)
training <- f[intrain,]
testing <- f[-intrain,]




# Set up cross-validation for training


ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)





#######Entropy Calculation
# Load the randomForest library
library(randomForest)

# Function to calculate entropy with the original formula
calculate_entropy_original <- function(labels) {
  n <- length(labels)
  if (n == 0) {
    return(0)
  }
  proportions <- table(labels) / n
  entropy <- -sum(proportions * log2(proportions), na.rm = TRUE)
  return(entropy)
}



# Load the dataset
data <- read.csv("Telco-Customer-Churn.csv")


# Calculate the entropy of the target variable using the original formula
entropy_original <- calculate_entropy_original(data$Churn)



# Print the entropy values
print(paste("Entropy of the target variable (Original formula):", entropy_original))



#---Decision Tree---#

# Train the decision tree model
tree_model <- train(Churn ~ ., data = training, method = "rpart",
                    trControl = ctrl, tuneLength = 10)
# Print the model details
print(tree_model)
# Plot the decision tree
library(rpart.plot)
rpart.plot(tree_model$finalModel, main = "Decision Tree")


# Make predictions on the test data
predictions1 <- predict(tree_model, newdata = testing)
# Evaluate the model performance
# Check unique levels in predictions and testing$Churn
unique_levels <- unique(c(levels(predictions1), levels(testing$Churn)))

# Relevel both variables with the unique levels
predictions <- factor(predictions1, levels = unique_levels)
testing$Churn <- factor(testing$Churn, levels = unique_levels)

# Check if the factor levels are now the same
identical(levels(predictions), levels(testing$Churn))

# If levels are identical, run confusionMatrix
confusionMatrix(predictions1, testing$Churn)





# Function to calculate entropy with the modified formula
calculate_entropy_modified <- function(labels) {
  n <- length(labels)
  if (n == 0) {
    return(0)
  }
  proportions <- table(labels) / n
  entropy <- -3.222 * sum(proportions * log10(proportions))
  return(entropy)
}

# Calculate the entropy of the target variable using the modified formula
entropy_modified <- calculate_entropy_modified(data$Churn)

# Print the entropy values for both original and modified formulas
print(paste("Entropy of the target variable (Modified formula):", entropy_modified))




#---Decision Tree---#

# Train the decision tree model
tree_model2 <- train(Churn ~ ., data = training, method = "rpart",
                     trControl = ctrl, tuneLength = 10)
# Print the model details
print(tree_model2)
# Plot the decision tree
library(rpart.plot)
rpart.plot(tree_model2$finalModel, main = "Decision Tree")


# Make predictions on the test data
predictions2 <- predict(tree_model2, newdata = testing)
# Evaluate the model performance
# Check unique levels in predictions and testing$Churn
unique_levels <- unique(c(levels(predictions2), levels(testing$Churn)))

# Relevel both variables with the unique levels
predictions3 <- factor(predictions2, levels = unique_levels)
testing$Churn <- factor(testing$Churn, levels = unique_levels)

# Check if the factor levels are now the same
identical(levels(predictions3), levels(testing$Churn))

# If levels are identical, run confusionMatrix
confusionMatrix(predictions3, testing$Churn)
