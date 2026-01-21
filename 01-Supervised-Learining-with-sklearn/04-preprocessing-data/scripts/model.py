# Import root_mean_squared_error
from sklearn.metrics import root_mean_squared_error

for name, model in models.items():
    # Fit the model to the scaled training data
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate the test RMSE
    test_rmse = root_mean_squared_error(y_test, y_pred)
    print("{} Test Set RMSE: {}".format(name, test_rmse))
