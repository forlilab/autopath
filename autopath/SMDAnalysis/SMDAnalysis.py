import pandas as pd
from typing import Optional
from autopath.SMDAnalysis import SMDData, PathModel

class SMDAnalysis:
    def __init__(
        self,
        data: SMDData,
        path_model: PathModel,
        # fe_model: FreeEnergyModel,
        # friction_model: FrictionModel,
    ):
        self.data = data
        self.path_model = path_model
        # self.fe_model = fe_model
        # self.friction_model = friction_model
        self.results: Optional[pd.DataFrame] = None
        return None