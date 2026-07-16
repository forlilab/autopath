import numpy as np, pandas as pd, pytest
from autopath.pulling.Estimators import recover_spring_constant, monotone_z_filter, FrictionEstimator

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

def test_monotone_z_filter():
    np.testing.assert_array_equal(monotone_z_filter(np.array([0.,1,2,3])), [True]*4)
    # folded step (index 2) dropped
    np.testing.assert_array_equal(monotone_z_filter(np.array([0.,1,0.5,2])), [True,True,False,True])
    # NaN dropped
    np.testing.assert_array_equal(monotone_z_filter(np.array([0.,np.nan,1.])), [True,False,True])

def test_per_path_feq_recovers_intercept():
    speeds=[0.005,0.01,0.015]; rows=[]
    for path,(feq0,gam) in {'p0':(100.,800.),'p1':(180.,600.)}.items():
        for step in range(6):
            feq=feq0+20*step
            for sp in speeds:
                rows.append(dict(path=path,step=step,speed=sp,r_coord=1.0+0.1*step,
                                 Fmean=feq+gam*sp))
    fr=pd.DataFrame(rows)
    out=FrictionEstimator().per_path_feq(fr)
    p0=out[out.path=='p0'].sort_values('step')
    np.testing.assert_allclose(p0['Feq'].to_numpy(), [100,120,140,160,180,200], atol=1.0)
    assert set(out.path)=={'p0','p1'}
