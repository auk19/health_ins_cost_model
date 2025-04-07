import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# process data
df = pd.read_csv("cleaned_ins.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns="cost")
y = df["cost"]

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# create and train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=10),
    "Gradient Boosting": GradientBoostingRegressor(random_state=10),
}

results = {}

# train and evaluate each model
for name, model in models.items():
    # train model
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluate model using MSE, RMSE, and R2 score
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}

    print(f"\n{name} Results:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

# save the best model
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")
