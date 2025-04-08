# HouseClassification

This project demonstrates how different categorical encoding techniques — OneHotEncoder, OrdinalEncoder, and TargetEncoder — impact the performance of a DecisionTreeClassifier when predicting housing price classes.

It uses a real-world dataset and evaluates models based on F1 Macro scores.  
The goal is to explore how preprocessing decisions affect final classification performance.

---

## Tools & Libraries Used

- pandas  
- scikit-learn (accuracy_score, classification_report, train_test_split, OneHotEncoder, OrdinalEncoder, DecisionTreeClassifier)  
- category_encoders (TargetEncoder)

---

## Model Results

| Encoder Type     | F1 Macro Score |
|------------------|----------------|
| OneHotEncoder    | 0.64           |
| OrdinalEncoder   | 0.86           |
| TargetEncoder    | 0.75           |

### What Do These Results Mean?

- F1 Macro Score measures the average F1-score across all classes equally, which is useful for imbalanced datasets.
- A higher F1 score indicates the model is performing better across all housing price categories.
- In this case:
  - OrdinalEncoder delivered the best overall performance.
  - TargetEncoder followed closely and is good for cases where labels relate directly to feature categories.
  - OneHotEncoder underperformed, likely due to high dimensionality from categorical expansion.

---

## Author

Eric Strzalkowski  
GitHub: [@Strzalaa](https://github.com/Strzalaa)

---