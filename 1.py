import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 1. Load Excel file (change 'your_file.xlsx' to your actual file name)
data = pd.read_excel("C:\\Users\\pc\\Desktop\\delhivery.csv")

# 2. Select features and target (modify these column names to match your Excel)
X = data[['osrm_distance', 'actual_distance_to_destination']]  # Example features
y = data['actual_time']  # Example target

# 3. Ridge Regression optimization
ridge = Ridge()
parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}  # λ values to test

# 4. Find best alpha using 3-fold cross-validation
grid = GridSearchCV(ridge, parameters, cv=3)
grid.fit(X, y)

# 5. Results
print("Best alpha (λ):", grid.best_params_['alpha'])
print("Cross-validation score (R²):", round(grid.best_score_, 3))

# 6. Make a sample prediction
sample_data = [[12.5, 10.2]]  # Replace with your values [osrm_distance, actual_distance]
prediction = grid.best_estimator_.predict(sample_data)
print(f"Predicted time: {prediction[0]:.1f} minutes")