import numpy as np, pandas as pd, pytest
from autopath.pulling.Estimators import recover_spring_constant

def test_recover_spring_constant_exact():
    k=7531.2
    rt=np.linspace(1.0,2.0,50); rb=rt-np.random.default_rng(0).normal(0,0.02,50)
    df=pd.DataFrame({'r_target':rt,'r_before':rb,'force':-k*(rb-rt)})
    assert recover_spring_constant(df)==pytest.approx(k, rel=1e-9)

def test_recover_spring_constant_ignores_bad_rows():
    k=9623.2
    df=pd.DataFrame({'r_target':[1.0,1.1,1.1],'r_before':[1.02,1.1,1.13],
                     'force':[-k*0.02, np.nan, -k*0.03]})
    assert recover_spring_constant(df)==pytest.approx(k, rel=1e-9)
