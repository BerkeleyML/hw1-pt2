from otter.test_files import test_case

OK_FORMAT = False

name = "q6b"
points = 2.0

@test_case(points=None, hidden=False)
def test_q6b_rmse_perfect(compute_rmse):
    import numpy as np
    y_true = np.array([1, 2, 3, 10, 12])
    y_pred = np.array([1, 2, 3, 10, 12])
    assert np.isclose(compute_rmse(y_true, y_pred), 0), 'RMSE should be 0 for perfect prediction'

@test_case(points=None, hidden=False)
def test_q6b_mae_perfect(compute_mae):
    import numpy as np
    y_true = np.array([1, 2, 3, 10, 12])
    y_pred = np.array([1, 2, 3, 10, 12])
    assert np.isclose(compute_mae(y_true, y_pred), 0), 'MAE should be 0 for perfect prediction'

@test_case(points=None, hidden=False)
def test_q6b_r2_perfect(compute_r2):
    import numpy as np
    y_true = np.array([1, 2, 3, 10, 12])
    y_pred = np.array([1, 2, 3, 10, 12])
    assert np.isclose(compute_r2(y_true, y_pred), 1), 'R2 should be 1 for perfect prediction'

