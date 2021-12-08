# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Ashwath Mohan created the model. It is a Gradient Boosted Classifier using the default hyperparameters in scikit-learn 1.0.1
Details of the model and implementations can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html').
## Intended Use
This model is intended for use in the Machine Learning Dev Ops Nanodegree. It should be used for educational purposes only.
## Training Data
The data used to train this model is the publicly available [Census Bureau data](https://www.kaggle.com/uciml/adult-census-income). 
The training data consists of 26048 examples, each of which has 105 features. This data can be recreated by setting the random_state to 42 
when making use of the [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 
function in scikit learn. 
## Evaluation Data
The data used to train this model is the publicly available [Census Bureau data](https://www.kaggle.com/uciml/adult-census-income). 
The testing data consistsof 6513 examples, each of which has 105 features. This data can be recreated by setting the random_state to 42 
when making use of the [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 
function in scikit learn. 
## Metrics
The following metrics were used to evaluate the model:
- precision: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
- recall: The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
The recall is intuitively the ability of the classifier to find all the positive samples.
- fbeta: The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.
- accuracy : Accuracy classification score, total correct/total examples
## Ethical Considerations
This data was publicly sourced. There might be inherent bias in the data that is not immediately obvious. One should be especially careful
in using results from this model to make judgements on certain groups of people and their income levels. 
## Caveats and Recommendations
Proper hyperparameter tuning could prove to be beneficial. Additionally, comparing performance across different models might allow 
for a higher score among the metrics defined above. 