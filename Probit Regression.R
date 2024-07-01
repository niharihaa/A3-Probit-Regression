# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)

# Read in the data
data <- read_csv("C:/Users/nihar/OneDrive/Desktop/Bootcamp/SCMA 632/DataSet/NSSO68.csv")

# Create the Target variable
data <- data %>%
  mutate(non_veg = ifelse(rowSums(select(., eggsno_q, fishprawn_q, goatmeat_q, beef_q, pork_q, chicken_q, othrbirds_q)) > 0, 1, 0))

# Get the value counts of non_veg
non_veg_values <- data$non_veg
value_counts <- table(non_veg_values)
print(value_counts)

# Interpretation: This tells us the distribution of non-vegetarians (1) and vegetarians (0) in the dataset.
# non_veg_values
#     0     1 
# 33072 68590 

# Ensure that the dataset contains both levels of the target variable
if (length(unique(data$non_veg)) < 2) {
  stop("The dataset does not contain both levels of the target variable 'non_veg'.")
}

# Define the dependent variable (non_veg) and independent variables
y <- data$non_veg
X <- data %>% 
  select(HH_type, Religion, Social_Group, Regular_salary_earner, Possess_ration_card, Sex, Age, Marital_Status, Education, Meals_At_Home, Region, hhdsz, NIC_2008, NCO_2004)

# Convert relevant columns to factors
y <- as.factor(y)
X <- X %>%
  mutate(
    Region = as.factor(Region),
    Social_Group = as.factor(Social_Group),
    Regular_salary_earner = as.factor(Regular_salary_earner),
    HH_type = as.factor(HH_type),
    Possess_ration_card = as.factor(Possess_ration_card),
    Sex = as.factor(Sex),
    Marital_Status = as.factor(Marital_Status),
    Education = as.factor(Education)
  )

# Combine the dependent and independent variables into one dataframe
combined_data <- data.frame(y, X)

# Inspect the combined data
str(combined_data)
head(combined_data)

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(combined_data$y, p = 0.8, list = FALSE)
train_data <- combined_data[train_index, ]
test_data <- combined_data[-train_index, ]

# Fit the probit regression model on the training data
probit_model <- glm(y ~ ., data = train_data, 
                    family = binomial(link = "probit"),
                    control = list(maxit = 1000))

# Print model summary
summary(probit_model)
# Interpretation: The summary provides the coefficients of the model. Significant variables have low p-values.
# The coefficients indicate the direction and magnitude of the relationship with the target variable.

# Predict probabilities on the test data
predicted_probs <- predict(probit_model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions using a threshold of 0.5
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

# Actual classes from the test data
actual_classes <- test_data$y

# Confusion Matrix
confusion_matrix <- confusionMatrix(as.factor(predicted_classes), as.factor(actual_classes))
print(confusion_matrix)
# Interpretation: The confusion matrix provides counts of true positives, true negatives, false positives, and false negatives.
# Sensitivity (Recall), Specificity, Positive Predictive Value (Precision), and Negative Predictive Value can be derived.

# ROC curve and AUC value
roc_curve <- roc(actual_classes, predicted_probs)
auc_value <- auc(roc_curve)
plot(roc_curve, col = "orange", main = "ROC Curve")
print(paste("AUC:", auc_value))
# Interpretation: The ROC curve shows the trade-off between sensitivity and specificity.
# AUC (Area Under the Curve) value indicates the model's ability to distinguish between classes. A higher AUC is better.
# In this case, AUC is 0.677, indicating moderate discriminatory ability.

# Accuracy, Precision, Recall, F1 Score
accuracy <- confusion_matrix$overall['Accuracy']
precision <- confusion_matrix$byClass['Pos Pred Value']
recall <- confusion_matrix$byClass['Sensitivity']
f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))
# Interpretation:
# - Accuracy: The proportion of correctly classified instances (70.7%).
# - Precision: The proportion of positive predictions that are actually correct (59.7%).
# - Recall (Sensitivity): The proportion of actual positives correctly identified by the model (18.8%).
# - F1 Score: The harmonic mean of precision and recall, balancing both (28.5%).