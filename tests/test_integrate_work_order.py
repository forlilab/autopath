import numpy as np
import pandas as pd

from autopath.pulling.SMDData import SMDData


def _logs():
    # two trajectories with overlapping index labels, as pd.concat of logs gives
    t = np.arange(5.0)
    a = pd.DataFrame({"trajname": "a", "time": t, "r_target": 1 + 0.1 * t,
                      "force": [1.0, 2.0, 3.0, 4.0, 5.0], "dW_protocol": [0.0, 1, 2, 3, 4]})
    b = a.assign(trajname="b", force=a["force"] * 10, dW_protocol=a["dW_protocol"] * 10)
    return pd.concat([a, b])


def test_work_follows_rows_when_logs_are_not_time_sorted():
    smd = SMDData.__new__(SMDData)
    smd.force_column = "force"
    ref = _logs()
    shuffled = ref.iloc[[3, 0, 4, 2, 1, 8, 5, 9, 7, 6]]
    for mode in ("force_dx", "protocol"):
        exp = smd.integrate_force_dx(ref.copy(), "r_target", work_mode=mode)
        got = smd.integrate_force_dx(shuffled.copy(), "r_target", work_mode=mode)
        key = ["trajname", "time"]
        exp, got = (d.sort_values(key).reset_index(drop=True) for d in (exp, got))
        np.testing.assert_allclose(got["work"], exp["work"])
