from otter.test_files import test_case

OK_FORMAT = False

name = "q9a"
points = 3.0

@test_case(points=None, hidden=False)
def test_q9a_model(env, model_rotation_regression):
    from sklearn.neural_network import MLPRegressor
    assert 'model_rotation_regression' in env, 'model_rotation_regression is not defined'
    assert isinstance(model_rotation_regression, MLPRegressor), 'model_rotation_regression is not an MLPRegressor'
    assert hasattr(model_rotation_regression, 'coefs_') and len(model_rotation_regression.coefs_) > 0, 'model_rotation_regression does not appear to be fitted'

