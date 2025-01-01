# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a RandomForestClassifier model, developed with the goal of making a categorical (binary in this case) prediction of whether an individual's income is greater than $50,000 or not. 

## Intended Use
This model is intended to make a binary "salary" prediction based on non-monetary features, such as Occupation, Age, Education level, and Marriage status

## Training Data
The model was trained on a random sampling of 80% of the 1994 US Census Income datasetin the UCI Machine Learning Repository. The dataset contains 15 total columns (features), and 32,561 rows (observations) 

## Evaluation Data
The model was evaluated on it's predictions of the the remaining 20% of the dataset, with the "salary" feature removed.

## Metrics
The Metrics used, and the model's scores are:
### Precision: 0.7419 | Recall: 0.6384 | F1-Score: 0.6863

## Ethical Considerations
Regarding the source of the data, although the dataset has been anonymized, personal data was still extracted and used. This could be seen as an invasion of privacy. Regarding the use of the data, one struggles to imagine a benificent use for this kind of prediction.

## Caveats and Recommendations
There are many other features that we could consider when making a prediction on income. Length of current employment, Technical training, and City/State of Residence are just a few examples of features that could be included to improve predictions and metrics.
