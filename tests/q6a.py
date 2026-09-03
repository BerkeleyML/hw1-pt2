from otter.test_files import test_case

OK_FORMAT = False

name = "q6a"
points = 2.0

@test_case(points=None, hidden=False)
def test_q6a_frame(prices_test, test_df):
    assert len(prices_test) == len(test_df), 'prices_test and test_df should have the same length'
    assert 'price_prediction' in prices_test.columns, 'price_prediction column not found in prices_test'

@test_case(points=None, hidden=False)
def test_q6a_model_type(price_model):
    from sklearn.linear_model import LinearRegression
    assert isinstance(price_model, LinearRegression), 'price_model is not a LinearRegression model'

