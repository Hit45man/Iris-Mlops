from sklearn.datasets import load_iris
import pandas as pd
import os

def save_iris_dataset(path='data/iris.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(path, index=False)
    print(f"Iris dataset saved to {path}")

if __name__ == "__main__":
    save_iris_dataset()








# from sklearn.datasets import load_iris
# import pandas as pd

# def save_data():
#     data = load_iris(as_frame=True)
#     df = data.frame
#     df.to_csv('data/iris.csv', index=False)

# if __name__ == "__main__":
#     save_data()
