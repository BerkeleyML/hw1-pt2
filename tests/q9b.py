from otter.test_files import test_case

OK_FORMAT = False

name = "q9b"
points = 3.0

@test_case(points=None, hidden=False)
def test_q9b_pred_length(y_pred_angles, X_test_rotated_sc):
    assert len(y_pred_angles) == len(X_test_rotated_sc), 'y_pred_angles should have one prediction per test example'

@test_case(points=None, hidden=False)
def test_q9b_mse(mse):
    assert isinstance(mse, float), 'mse should be a float'
    assert mse >= 0, 'mse should be non-negative'

@test_case(points=None, hidden=False)
def test_q9b_rmse(rmse):
    assert isinstance(rmse, float), 'rmse should be a float'
    assert rmse >= 0, 'rmse should be non-negative'

