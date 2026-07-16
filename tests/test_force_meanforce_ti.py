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

def test_meanforce_ti_pmf_two_path():
    import numpy as np, pandas as pd
    from autopath.pulling.Estimators import FrictionEstimator
    speeds=[0.005,0.01,0.015]; k=9623.2; rows=[]
    # monotone-in-lambda, well-behaved (dFeq/dlam << k) so z stays monotone
    for path,feq0 in {'p0':80.,'p1':120.}.items():
        for step in range(20):
            lam=1.0+0.05*step; feq=feq0+50*step          # dFeq/dlam=1000 << k
            for sp in speeds:
                rows.append(dict(path=path,step=step,speed=sp,r_coord=lam,Fmean=feq+700*sp))
    fr=pd.DataFrame(rows)
    w={sp:{'p0':0.5,'p1':0.5} for sp in speeds}
    mix,diag=FrictionEstimator().meanforce_ti_pmf(fr,w,k)
    assert mix is not None and {'z','dG_z'} <= set(mix.columns)
    assert mix['dG_z'].iloc[-1] > mix['dG_z'].iloc[0]      # rises outward
    assert diag['fell_back_paths']==[]                    # z stayed monotone
    assert diag['z_divergence_nm_max'] > 0                # paths differ

def test_meanforce_ti_pmf_folded_falls_back(caplog):
    import numpy as np, pandas as pd
    from autopath.pulling.Estimators import FrictionEstimator
    speeds=[0.005,0.01,0.015]; k=1000.0; rows=[]   # tiny k -> z folds
    for step in range(20):
        lam=1.0+0.05*step; feq=100.+5000*step             # dFeq/dlam=1e5 >> k
        for sp in speeds:
            rows.append(dict(path='p0',step=step,speed=sp,r_coord=lam,Fmean=feq+700*sp))
    fr=pd.DataFrame(rows); w={sp:{'p0':1.0} for sp in speeds}
    mix,diag=FrictionEstimator().meanforce_ti_pmf(fr,w,k)
    assert 'p0' in diag['fell_back_paths']                 # fell back to lambda, no silent sort

def test_meanforce_ti_pmf_sign_guard_raises():
    speeds = [0.005, 0.01, 0.015]
    rows = []
    for step in range(10):
        feq = -(100.0 + 10 * step)          # NEGATIVE mean force -> sign convention violated
        for sp in speeds:
            rows.append(dict(path='p0', step=step, speed=sp,
                             r_coord=1.0 + 0.05 * step, Fmean=feq + 700 * sp))
    fr = pd.DataFrame(rows)
    w = {sp: {'p0': 1.0} for sp in speeds}
    with pytest.raises(ValueError, match="sign convention"):
        FrictionEstimator().meanforce_ti_pmf(fr, w, 9623.2)


def test_meanforce_ti_skipped_single_speed():
    # meanforce_ti_pmf returns (None, {}) when <2 speeds -> per_path_feq empty
    import pandas as pd
    from autopath.pulling.Estimators import FrictionEstimator
    fr=pd.DataFrame([dict(path='p0',step=s,speed=0.01,r_coord=1.0+0.05*s,Fmean=100.+10*s) for s in range(10)])
    mix,diag=FrictionEstimator().meanforce_ti_pmf(fr,{0.01:{'p0':1.0}},9623.2)
    assert mix is None
