from otter.test_files import test_case

OK_FORMAT = False

name = "q10"
points = 5.0

@test_case(points=0.0, hidden=False)
def test_q10_callable(test_time_augmentation):
    assert callable(test_time_augmentation), 'do not rename the test_time_augmentation function'

