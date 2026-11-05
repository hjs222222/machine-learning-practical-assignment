import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def metric_rows(strategy, model, truth, prediction, first_h=1):
    truth = np.asarray(truth).reshape(len(truth),-1)
    prediction = np.asarray(prediction).reshape(truth.shape)
    return [dict(strategy=strategy,model=model,h=first_h+j,n=len(truth),
                 R2=float(r2_score(truth[:,j],prediction[:,j])),
                 RMSE=float(np.sqrt(mean_squared_error(truth[:,j],prediction[:,j]))),
                 MAE=float(mean_absolute_error(truth[:,j],prediction[:,j]))) for j in range(truth.shape[1])]
