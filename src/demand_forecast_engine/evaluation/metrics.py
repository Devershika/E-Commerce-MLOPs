import numpy as np

class ForecastingMetrics:
    """
    A utility class to calculate common forecasting accuracy metrics.

    This class provides methods to evaluate the performance of a forecasting model
    by comparing actual values (y_true) against predicted values (y_pred).

    Attributes:
        y_true (np.ndarray): Array of actual ground truth values.
        y_pred (np.ndarray): Array of predicted values from the model.
    """

    def __init__(self, y_true, y_pred):
        """
        Initializes ForecastingMetrics with true and predicted values.

        Args:
            y_true (list or np.ndarray): Actual observation values.
            y_pred (list or np.ndarray): Predicted values.
        """
        super(ForecastingMetrics, self).__init__()
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)

    def MAPE(self):
        """
        Calculates the Mean Absolute Percentage Error (MAPE).

        MAPE measures the average magnitude of error produced by a model 
        relative to the actual values, expressed as a percentage. 
        Zero values in y_true are masked to avoid division by zero errors.

        Formula:
        $$MAPE = \frac{100\%}{n} \sum_{t=1}^{n} \left| \frac{y_{true} - y_{pred}}{y_{true}} \right|$$

        Returns:
            float: The MAPE value rounded to 2 decimal places.
        """
        mask = self.y_true != 0
        mape_val = np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
        return mape_val.round(2)
    
    def WMAPE(self):
        """
        Calculates the Weighted Mean Absolute Percentage Error (WMAPE).

        WMAPE (also known as MAD/Mean ratio) weights the absolute errors 
        by the volume of actuals. This is particularly useful in demand 
        forecasting to avoid the skewing effect of low-volume items.

        Formula:
        $$WMAPE = \frac{\sum |y_{true} - y_{pred}|}{\sum |y_{true}|} \times 100$$

        Returns:
            float: The WMAPE value rounded to 2 decimal places.

        Raises:
            ValueError: If the sum of the absolute actual values is zero.
        """
        total_abs_error = np.sum(np.abs(self.y_pred - self.y_true))
        total_actuals_sum = np.sum(np.abs(self.y_true))
        if total_actuals_sum == 0.0:
            raise ValueError("Total actuals is zero! Cannot divide by zero for WMAPE.")
        wmape_val = (total_abs_error / total_actuals_sum) * 100
        return wmape_val.round(2)