from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PredictedMeasurement:
    # lm_key: gtsam.Key    # or lm_id: int
    lm_id: int             # useful for graph update / logging
    zbar: np.ndarray       # shape (2,) -> [range, bearing]