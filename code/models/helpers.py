from sklearn import metrics
import numpy as np
from datetime import datetime

def root_mean_squared_log_error(y, y_pred):
    y_pred[y_pred < 0] = 0 # Some predictions are negative, that screws up the log
    return -metrics.mean_squared_error(np.log(y + 1), np.log(y_pred + 1)) ** 0.5

rmsle_scorer = metrics.make_scorer(root_mean_squared_log_error)

# calculates time taken by a model (to compare time complexity)
# call it before running a model: start=timer()
# print time taken after model:   time(start)
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def root_mean_squared_log_error_minutes(regressor, X,y):
    y_pred = regressor.predict(X)/60
    y_pred=np.round(y_pred)*60//60
    y=y/60
    y=y.round()
    y=y*60
    y=y//60
    y_pred[y_pred < 0] = 0 # Some predictions are negative, that screws up the log
    print("RMSLE for minutes", metrics.mean_squared_error(np.log(y + 1), np.log(y_pred + 1)) ** 0.5)
