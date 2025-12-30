from autopath.SMDAnalysis.SMDData import SMDData
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """
    Base class for all estimators in SMDAnalysis.
    """

    def __init__(self):
        pass
    
    def fit(self, smd_data: SMDData):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, smd_data: SMDData):
        raise NotImplementedError("Subclasses should implement this method.")
    
class JarzynskiEstimator(BaseEstimator):
    def fit(self, smd_data: SMDData):
        # Implement fitting logic specific to Jarzynski estimator
        pass