from otter.test_files import test_case

OK_FORMAT = False

name = "q8b"
points = 3.0

@test_case(points=None, hidden=False)
def test_q8b_model(env, model_rotated):
    from sklearn.neural_network import MLPClassifier
    assert 'model_rotated' in env, 'model_rotated is not defined'
    assert isinstance(model_rotated, MLPClassifier), 'model_rotated is not an MLPClassifier'
    assert hasattr(model_rotated, 'coefs_') and len(model_rotated.coefs_) > 0, 'model_rotated does not appear to be fitted'

