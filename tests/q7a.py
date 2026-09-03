from otter.test_files import test_case

OK_FORMAT = False

name = "q7a"
points = 3.0

@test_case(points=None, hidden=False)
def test_q7a_type(env, test_secret_df):
    import pandas as pd
    assert 'test_secret_df' in env, 'test_secret_df is not defined'
    assert isinstance(test_secret_df, pd.DataFrame), 'test_secret_df should be a DataFrame'

@test_case(points=None, hidden=False)
def test_q7a_frame(test_secret_df):
    import numpy as np
    assert len(test_secret_df) == 2000, 'test_secret_df should have 2000 rows'
    assert 'correct' in test_secret_df.columns, 'correct column is missing from test_secret_df'
    assert 'secret_idx' in test_secret_df.columns, 'secret_idx column is missing from test_secret_df'
    assert np.array_equal(np.sort(test_secret_df['secret_idx'].to_numpy()), np.arange(len(test_secret_df))), 'secret_idx should identify each row of X_test_secret'

