import numpy as np


class SimpleLR:
    __doc__ = """
    Simple Linear Regression

    This model fits a simple linear regression to the provided data. It is a straightforward example of how to implement custom behavior in a class for the purpose of fitting a model, making predictions, and generating a summary.

    Parameters
    ----------
    exog : array_like
        The input data features for model fitting. Should be a 2D array.
    y : array_like
        The target variable for model fitting. Should be a 1D array.

    Attributes
    ----------
    coef_ : float
        The slope of the linear model.
    intercept_ : float
        The intercept of the linear model.

    Examples
    --------
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> model = SimpleLinearRegression(X, y)
    >>> model.fit()
    >>> print(model.summary())
    >>> predictions = model.predict(np.array([[6], [7]]))
    >>> print(predictions)
    """

    def __init__(self, exog, y):
        if isinstance(exog, list):  # Check if X is a list
            # Convert X to a NumPy array if it's a list
            self.exog = np.array(exog)
        else:
            self.exog = exog

        if isinstance(y, list):  # Check if y is a list
            self.y = np.array(y)  # Convert y to a NumPy array if it's a list
        else:
            self.y = y

        self.coef_ = None
        self.intercept_ = None

    def fit(self):
        """
        Fit the simple linear regression model to the data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Ensure X is a 2D array
        if self.exog.ndim == 1:
            self.exog = self.exog[:, np.newaxis]
        X_with_intercept = np.c_[np.ones((self.exog.shape[0], 1)), self.exog]

        # Normal Equation Method to find coefficients
        weights = np.linalg.inv(X_with_intercept.T @
                                X_with_intercept) @ X_with_intercept.T @ self.y

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

        return self

    def summary(self):
        """
        Generate a summary of the fitted model.

        Returns
        -------
        summary : str
            Summary of the model's fit.
        """
        summary = f"Model Summary:\nIntercept: {self.intercept_}\nCoefficient(s): {self.coef_}"
        return summary

    def predict(self, X_new):
        """
        Predict using the simple linear regression model.

        Parameters
        ----------
        X_new : array_like
            The new input data features for which to make predictions. Should be a 2D array.

        Returns
        -------
        predictions : array_like
            Predictions for each input data row.
        """
        if isinstance(X_new, list):
            X_new = np.array(X_new)

        if X_new.ndim == 1:
            X_new = X_new[:, np.newaxis]
        return self.intercept_ + np.dot(X_new, self.coef_)


'''
Model Class Structure and Documentation
1. Class Definition and Initialization
__init__ method: The class must include an __init__ method that accepts input features (exog) parameters. The __init__ method can also include optional parameters with default values.

Parameters:
exog: array_like, the explanatory variables of the model. Should be a 2D array for multiple features.


The class should initialize necessary attributes, such as model coefficients (coef_) and intercept (intercept_), setting them to None or appropriate default values.
2. Fitting the Model
fit method: A method that performs the model fitting process using the provided exog and y. This method calculates and sets the model parameters, such as coef_ and intercept_.

Returns: The fit method should return the instance itself (self), allowing for method chaining.
3. Model Summary
summary method: A method that returns a summary of the fitted model, including important statistics, model coefficients, and performance metrics.

Returns: A string containing the summary information.
4. Making Predictions
predict method: A method that accepts new input features (X_new) and returns predictions based on the model.

Parameters:

X_new: array_like, new input data for which to make predictions. Should be a 2D array for multiple features.
Returns: An array_like object containing the predictions for each input row.

5. Documentation
__doc__ string: A detailed documentation string at the beginning of the class definition providing an overview of the model, its parameters, methods, and usage examples.

The format of the documentation should follow the statsmodel docstring conventions, including a brief description, parameters, attributes, examples, and usage instructions.

Include a brief description of the model.
List and explain parameters accepted by __init__.
Describe the purpose and use of each method (fit, summary, predict).
Provide example usage demonstrating how to initialize the model, fit it, print a summary, and make predictions.


'''
