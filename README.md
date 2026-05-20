## Group Members
- RAJA AHMAD AFIQ BIN RAJA NASARUDDIN [BI23110390]
- NICOLE VIVIENNE VOO [BI23110020]
- GLORIA ASTEFEN JANE GUDIN  [BI23110080]
- VINOTHINI A/P CHANDRA MOHAN  [BI23110287]
- THESVEND A/L RAMESH  [BI23160441]  


## Milestone 4: Model Optimization

This milestone focuses on improving the model and reducing overfitting. Since the target variable, Performance, is categorical, a classification approach was used for model optimization.

A baseline Decision Tree model was first trained without parameter tuning. This model may overfit because it can grow too deeply and learn patterns that are too specific to the training data.

To reduce overfitting, GridSearchCV was used to tune important Decision Tree parameters such as max_depth, min_samples_split, min_samples_leaf, and criterion. These parameters help control the complexity of the model.

The optimized model was compared with the baseline model using training accuracy, testing accuracy, classification report, confusion matrix, and overfitting gap. A smaller gap between training and testing accuracy indicates that the model generalizes better to unseen data.
