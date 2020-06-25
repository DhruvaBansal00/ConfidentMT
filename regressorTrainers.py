from sklearn.linear_model import LogisticRegression

def trainLogisticRegressor(X, Y, verbose=False):
    currReg = LogisticRegression(random_state=42, fit_intercept=True, solver='saga', max_iter=1000,
        multi_class='multinomial', n_jobs=-1).fit(X, Y)
    return currReg

def getTrainerFromRegressor(regressor):
    string_to_trainer = {'LogisticRegressor': trainLogisticRegressor}
    return string_to_trainer[regressor]