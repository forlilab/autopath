"""On-demand metadynamics summary: data, analysis, figures, convergence checks."""
from .data import load_run
from .report import report
from .convergence import check_convergence, estimators_agree
