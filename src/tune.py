from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from preprocess import load_and_split

def tune():
    X_train, _, y_train, _ = load_and_split()
    grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10]
    }
    model = RandomForestClassifier()
    search = GridSearchCV(model, grid, cv=3)
    search.fit(X_train, y_train)
    print("Best Params:", search.best_params_)

if __name__ == "__main__":
    tune()
