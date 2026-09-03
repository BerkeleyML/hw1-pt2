from otter.test_files import test_case

OK_FORMAT = False

name = "q7ci"
points = 0.5

@test_case(points=None, hidden=False)
def test_q7ci_type(env, conf_matrix, predictions, y_test_secret):
    import numpy as np
    assert 'predictions' in env, 'predictions is not defined'
    assert 'conf_matrix' in env, 'conf_matrix is not defined'
    assert len(predictions) == len(y_test_secret), 'predictions should have one entry per secret test example'
    assert isinstance(conf_matrix, np.ndarray), 'conf_matrix should be a numpy array'
    assert np.issubdtype(conf_matrix.dtype, np.integer), 'conf_matrix should contain integer values'

@test_case(points=None, hidden=False)
def test_q7ci_shape_and_sum(conf_matrix, model, y_test_secret):
    assert conf_matrix.shape == (len(model.classes_), len(model.classes_)), 'conf_matrix should be a square matrix with the number of classes'
    assert conf_matrix.sum() == len(y_test_secret), 'sum of values in the confusion matrix should be the number of samples in the test set'

