from otter.test_files import test_case

OK_FORMAT = False

name = "q6d"
points = 2.0

@test_case(points=None, hidden=False)
def test_q6d_misclassified(env, misclassified_price_error):
    assert 'misclassified_price_error' in env, 'misclassified_price_error is not defined'
    assert isinstance(misclassified_price_error, float), 'misclassified_price_error is not a float'

@test_case(points=None, hidden=False)
def test_q6d_correctly_classified(env, correctly_classified_price_error):
    assert 'correctly_classified_price_error' in env, 'correctly_classified_price_error is not defined'
    assert isinstance(correctly_classified_price_error, float), 'correctly_classified_price_error is not a float'

