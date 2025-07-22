import pandas as pd
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Excel data
data = pd.read_excel("C:\\Users\\pc\\Desktop\\delhivery.csv")  # Replace with your file path

# 2. Select features and target
X = data[['osrm_distance', 'actual_distance_to_destination']]  # Features
y = data['actual_time']  # Target

# 3. Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Linear SVR with hyperparameter tuning (using only training data)
svr = LinearSVR(random_state=42)
parameters = {
    'epsilon': [0.1, 0.5, 1.0],
    'C': [0.1, 1, 10],
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
}

# 5. Grid search with 3-fold cross-validation (on training data)
grid = GridSearchCV(svr, parameters, cv=3, scoring='r2')
grid.fit(X_train, y_train)

# 6. Evaluate on the test set (unseen data)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# 7. Calculate accuracy metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n=== Model Performance ===")
print("Best Parameters:", grid.best_params_)
print(f"Test R² Score: {r2:.3f} (1.0 is perfect)")
print(f"Mean Absolute Error (MAE): {mae:.3f} minutes")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} minutes")

# 8. Sample prediction
sample = [[15.0, 12.3]]  # Example input
print("\nSample Prediction:", best_model.predict(sample)[0], "minutes")