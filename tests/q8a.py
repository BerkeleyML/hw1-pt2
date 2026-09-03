from otter.test_files import test_case

OK_FORMAT = False

name = "q8a"
points = 3.0

@test_case(points=None, hidden=False)
def test_q8a_types(X_train_rotated, y_train_augmented):
    import numpy as np
    assert isinstance(X_train_rotated, np.ndarray), 'X_train_rotated should be a numpy array'
    assert isinstance(y_train_augmented, np.ndarray), 'y_train_augmented should be a numpy array'

@test_case(points=None, hidden=False)
def test_q8a_shapes(X_train_rotated, num_rotations_per_image):
    assert X_train_rotated.shape[1] == 784, 'X_train_rotated should have 784 features'
    assert X_train_rotated.shape[0] == 3000 * num_rotations_per_image, 'X_train_rotated should have 3000 * num_rotations_per_image rows'

