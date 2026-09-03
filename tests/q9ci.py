from otter.test_files import test_case

OK_FORMAT = False

name = "q9ci"
points = 3.0

@test_case(points=None, hidden=False)
def test_q9ci_scaled_shape(X_test_secret_scaled, X_test_secret):
    assert X_test_secret_scaled.shape == X_test_secret.shape, 'X_test_secret_scaled should have the same shape as X_test_secret'

@test_case(points=None, hidden=False)
def test_q9ci_scaled_values(X_test_secret_scaled, X_test_secret, scaler):
    import numpy as np
    assert hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'), 'scaler does not appear to be fitted'
    assert np.isfinite(X_test_secret_scaled).all(), 'X_test_secret_scaled should contain only finite values'
    _expected = (X_test_secret - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(X_test_secret_scaled, _expected, rtol=1e-06, atol=1e-06, err_msg='X_test_secret_scaled does not match the fitted scaler')

@test_case(points=None, hidden=False)
def test_q9ci_angles_type(y_pred_angles_secret):
    import numpy as np
    assert isinstance(y_pred_angles_secret, np.ndarray), 'y_pred_angles_secret should be a numpy array'
    assert y_pred_angles_secret.dtype == np.float64, 'y_pred_angles_secret should have dtype float64'

@test_case(points=None, hidden=False)
def test_q9ci_angles_shape(y_pred_angles_secret, X_test_secret):
    assert y_pred_angles_secret.shape == (len(X_test_secret),), 'y_pred_angles_secret should have one angle per secret test example'

