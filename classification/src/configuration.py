import numpy as np

DELIMITER = ','

PREDICTOR_COLUMN_DTYPE = np.float32
LABEL_COLUMN_DTYPE = 'category'
GROUP_COLUMN_DTYPE = str
CELL_ID_COLUMN_DTYPE = np.int64

RAW_NAN_REPRESENTATIONS = {np.NAN, np.nan, np.NaN, None}
RAW_PINF_REPRESENTATIONS = {float('inf'), np.inf, np.infty, np.Inf, np.Infinity, np.PINF}
RAW_NINF_REPRESENTATIONS = {float('-inf'), -np.inf, -np.infty, -np.Inf, -np.Infinity, np.NINF}

NAN_REPRESENTATION = np.nan
PINF_REPRESENTATION = np.finfo(PREDICTOR_COLUMN_DTYPE).max
NINF_REPRESENTATION = np.finfo(PREDICTOR_COLUMN_DTYPE).min

# GROUP_COLUMN_NAME = 'image'  # same image = same patient
# LABEL_COLUMN_NAME = 'label'
# CELL_ID_COLUMN_NAME = 'cell_id'  # cell ID within an image

NUM_BAYESIAN_OPTIMIZATION_ITERATIONS = 25
