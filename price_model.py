import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Initialize MLflow experiment
mlflow.set_experiment("HousePricePrediction")

with mlflow.start_run():
    # Define the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("mae...."+ str(mae))
    print("mse...." + str(mse))
    print("r2...." + str(r2))
    # Log metrics to MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2_Score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "linear_regression_model")

    # Save trained model
    joblib.dump(model, "house_price_model.pkl")

print("✅ Model training complete. Model saved as 'house_price_model.pkl'.")
