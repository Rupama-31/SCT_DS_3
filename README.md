Predictive Modeling of Financial Consumer Behavior

SkillCraft Technology - Task 03

1.Project Overview
This project uses a Decision Tree Classifier to predict whether a customer will subscribe to a bank term deposit based on a combination of demographic, behavioral, and macroeconomic data. As an economics-focused analysis, it explores how market indicators influence individual financial decisions.
 
2.Key Findings & Economic Interpretation
The model achieved a 91.50% prediction accuracy. Below are the most significant determinants found by the model:
Duration (0.496): This was the strongest predictor. Economically, this suggests that reducing Information Asymmetry through longer customer engagement significantly increases the likelihood of a transaction.
Number of Employees (0.360): A proxy for Macroeconomic Stability. This indicates that consumer propensity to save is highly correlated with the overall health of the labor market.
Euribor 3M Rate (0.043): This represents the Interest Rate Environment. It proves that consumers are sensitive to the "opportunity cost" of capital when choosing bank products.
Consumer Confidence Index (0.033): Highlighting the role of Expectation Theory, where the "Animal Spirits" or future outlook of the consumer determines their current savings rate.

3.🛠️ Tools & Libraries
Python: Primary programming language.  
Pandas: For data cleaning and handling semicolon-separated values.  
Scikit-Learn: For building the Decision Tree and performing the train-test split.  
Matplotlib: Used for visualizing the decision logic flowchart.

4.Project Structure
main.py: The full Python script including data cleaning and visualization.
bank-additional-full.csv: The UCI Bank Marketing Dataset.  
bank_decision_tree.png: The visual representation of the model's logic.
