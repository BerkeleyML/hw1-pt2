from otter.test_files import test_case

OK_FORMAT = False

name = "q9cii"
points = 2.0

@test_case(points=None, hidden=False)
def test_q9cii_columns(test_secret_df):
    assert 'unrotated_prediction' in test_secret_df.columns, 'unrotated_prediction column is missing from test_secret_df'
    assert 'unrotated_correct' in test_secret_df.columns, 'unrotated_correct column is missing from test_secret_df'
    assert test_secret_df['unrotated_correct'].dtype == bool, 'unrotated_correct should have dtype bool'

