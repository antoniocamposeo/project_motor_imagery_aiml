from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def classification(X, Y, step, param_space, cv):
    """
    :param X: np_array
    :param Y: np_array
    :param step: dict - Pipeline steps
    :param param_space: dict - space of parameter to evaluate
    :param cv: int - k fold split
    :return: grid: object of GridSearch
    :return: train_score, test_score:
    :return y_test, y_pred:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=42, shuffle=Y)
    pipe = Pipeline(steps=step)
    grid = GridSearchCV(estimator=pipe, param_grid=param_space,
                        cv=cv, n_jobs=-1, verbose=3, return_train_score=True)
    grid.fit(X_train, y_train)
    train_score = grid.score(X_train, y_train)
    test_score = grid.score(X_test, y_test)
    y_pred = grid.predict(X_test)
    return grid, train_score, test_score, y_test, y_pred


def classification_TPOT(X, Y, TPot):
    """
    Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming
    :param TPot: object of TPOT Classifier
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=42, shuffle=Y)

    TPot.fit(X_train, y_train)
    y_pred = TPot.predict(X_test)
    train_score = TPot.score(X_train, y_train)
    test_score = TPot.score(X_test, y_test)

    return TPot, train_score, test_score, y_test, y_pred


