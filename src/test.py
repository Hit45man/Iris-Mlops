import pandas as pd

def test_data_shape():
    df = pd.read_csv('data/iris.csv')
    assert df.shape[1] == 5  # 4 features + target

if __name__ == "__main__":
    test_data_shape()
