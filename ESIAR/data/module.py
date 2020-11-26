import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from pandas import DataFrame
import numpy as np
import pylab
from matplotlib.widgets import Slider


# main
def corr_matrix(dt):
    sns.heatmap(dt.corr(), cbar=True, annot=True)
    return plt.show()


def describe(dt):
    return dt.describe()


# ridge
def grid_search_ridge(xvar, yvar, alphas, degrees, cv):
    alphas = list(map(float, alphas.split(',')))
    degrees = list(map(int, degrees.split(',')))
    cv = int(cv)
    param_grid = [
        {
            'ridge__alpha': alphas,
            'polynomialfeatures__degree': degrees
        },
    ]
    reg_d = make_pipeline(
        PolynomialFeatures(include_bias=False),
        StandardScaler(),
        Ridge()
    )
    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)
    grid_search = GridSearchCV(reg_d, param_grid, n_jobs=-1, cv=ss, scoring='r2')
    grid_search.fit(xvar, yvar)
    return list(grid_search.best_params_.items())[0][1], list(grid_search.best_params_.items())[1][1]


def ridge_model(alpha, degree):
    alpha = float(alpha)
    degree = int(degree)
    reg = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha)
    )
    return reg


def test_train_cv(xvar, yvar, model):
    b, b1, tmp, dt = [], [], {}, DataFrame()
    model1 = model
    for i in range(10):
        xn, xtest, yn, ytest = train_test_split(xvar, yvar)
        model1.fit(xn, yn)
        b.append(r2_score(ytest, model1.predict(xtest)))
        b1.append(mean_squared_error(ytest, model1.predict(xtest)))
    cv_r2, cv_mse = sum(b) / len(b), sum(b1) / len(b1)
    model.fit(xvar, yvar)
    tmp[''] = 'Train'
    tmp['MSE'] = mean_squared_error(yvar, model.predict(xvar))
    tmp['R2'] = r2_score(yvar, model.predict(xvar))
    dt = dt.append([tmp])
    tmp[''] = 'CV'
    tmp['MSE'] = cv_mse
    tmp['R2'] = cv_r2
    dt = dt.append([tmp])
    dt.set_index('', inplace=True)
    return dt


def graph_val_ridge_alpha(xvar, yvar, degree, alphas, cv):
    degree = int(degree)
    alphas = list(map(float, alphas.split(',')))
    cv = int(cv)
    param_grid = [{'ridge__alpha': alphas}]


    reg = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge())
    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)
    grid_search = GridSearchCV(reg, param_grid, cv=ss, scoring='r2', return_train_score=True)
    grid_search.fit(xvar, yvar)
    cv_res = DataFrame(grid_search.cv_results_)
    cv_res = cv_res.rename(
        columns={'param_ridge__alpha': 'alpha', 'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})
    tabl_alpha = cv_res[['alpha', 'mse_te', 'std_test_score', 'mse_tr', 'std_train_score']]
    tabl_alpha = tabl_alpha.assign(mean_test_score=cv_res.mse_te)
    tabl_alpha = tabl_alpha.assign(mean_train_score=cv_res.mse_tr)
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
    tabl_alpha.plot.bar(x='alpha', y='mean_test_score', ax=axes[0][0], color='red').grid(linewidth=0.2)
    tabl_alpha.plot.bar(x='alpha', y='std_test_score', ax=axes[0][1], color='red').grid(linewidth=0.2)
    tabl_alpha.plot.bar(x='alpha', y='mean_train_score', ax=axes[1][0]).grid(linewidth=0.2)
    tabl_alpha.plot.bar(x='alpha', y='std_train_score', ax=axes[1][1]).grid(linewidth=0.2)
    return plt.show()


def graph_val_ridge_degree(xvar, yvar, degrees, alpha, cv):
    degrees = list(map(int, degrees.split(',')))
    alpha = float(alpha)
    cv = int(cv)
    param_grid = [{'polynomialfeatures__degree': degrees}]

    reg1 = make_pipeline(
        PolynomialFeatures(include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha))
    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)
    grid_search1 = GridSearchCV(reg1, param_grid, cv=ss, scoring='r2', return_train_score=True)
    grid_search1.fit(xvar, yvar)

    cv_res1 = DataFrame(grid_search1.cv_results_)
    cv_res1 = cv_res1.rename(columns={'param_polynomialfeatures__degree': 'degree',
                                      'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})
    tabl_degree = cv_res1[['degree', 'mse_te', 'std_test_score', 'mse_tr', 'std_train_score']]
    tabl_degree = tabl_degree.assign(mean_test_score=cv_res1.mse_te)
    tabl_degree = tabl_degree.assign(mean_train_score=cv_res1.mse_tr)
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
    tabl_degree.plot.bar(x='degree', y='mean_test_score', ax=axes[0][0], color='red').grid(linewidth=0.2)
    tabl_degree.plot.bar(x='degree', y='std_test_score', ax=axes[0][1], color='red').grid(linewidth=0.2)
    tabl_degree.plot.bar(x='degree', y='mean_train_score', ax=axes[1][0]).grid(linewidth=0.2)
    tabl_degree.plot.bar(x='degree', y='std_train_score', ax=axes[1][1]).grid(linewidth=0.2)
    return plt.show()


def r_model_pridict(arg, model):
    x_matrix = np.array([arg]).reshape(1, -1)
    return model.predict(x_matrix)


# graph_data

def graph_data_4var(x_lable, x_min, x_max, flag, model, y_var,
                    one_lable, one_min, one_max,
                    two_lable, two_min, two_max,
                    three_lable, three_min, three_max):
    def gauss1(arg, x):
        arg.insert(flag, -1)
        s = []
        for i in x:
            arg = arg.copy()
            arg[flag] = i
            s.append(r_model_pridict(arg, model)[0])
        return s

    def updateGraph(one, two, three, graph_axes):
        al = [one, two, three]
        x = np.arange(x_min, x_max, 0.5)
        y = gauss1(al, x=x)
        graph_axes.clear()
        graph_axes.plot(x, y)
        graph_axes.legend(y_var, fontsize=8, ncol=2)
        graph_axes.grid(linewidth=0.2)
        graph_axes.set_xlabel(x_lable)
        # graph_axes.set(ylim=(0, 3))
        pylab.draw()

    def onChangeValue(values):
        updateGraph(slider_one.val, slider_two.val, slider_three.val, graph_axes)

    fig, graph_axes = pylab.subplots()
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.4)

    axes_slider_one = pylab.axes([0.07, 0.25, 0.85, 0.04])
    slider_one = Slider(axes_slider_one,
                        label=one_lable, valmin=one_min, valmax=one_max, valinit=(one_max + one_min) / 2,
                        valfmt='%1.1f')
    slider_one.on_changed(onChangeValue)

    axes_slider_two = pylab.axes([0.07, 0.17, 0.85, 0.04])
    slider_two = Slider(axes_slider_two,
                        label=two_lable, valmin=two_min, valmax=two_max, valinit=(two_max + two_min) / 2,
                        valfmt='%1.1f')
    slider_two.on_changed(onChangeValue)

    axes_slider_three = pylab.axes([0.07, 0.10, 0.85, 0.04])
    slider_three = Slider(axes_slider_three,
                          label=three_lable, valmin=three_min, valmax=three_max, valinit=(three_max + three_min) / 2,
                          valfmt='%1.1f')
    slider_three.on_changed(onChangeValue)

    updateGraph(slider_one.val, slider_two.val, slider_three.val, graph_axes)
    pylab.show()


def graph_data_3var(x_lable, x_min, x_max, flag, model, y_var,
                    one_lable, one_min, one_max,
                    two_lable, two_min, two_max):
    def gauss1(arg, x):
        arg.insert(flag, -1)
        s = []
        for i in x:
            arg = arg.copy()
            arg[flag] = i
            s.append(r_model_pridict(arg, model)[0])
        return s

    def updateGraph(one, two, graph_axes):
        al = [one, two]
        x = np.arange(x_min, x_max, 0.5)
        y = gauss1(al, x=x)
        graph_axes.clear()
        graph_axes.plot(x, y)
        graph_axes.legend(y_var, fontsize=8, ncol=2)
        graph_axes.grid(linewidth=0.2)
        graph_axes.set_xlabel(x_lable)
        # graph_axes.set(ylim=(0, 3))
        pylab.draw()

    def onChangeValue(values):
        updateGraph(slider_one.val, slider_two.val, graph_axes)

    fig, graph_axes = pylab.subplots()
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.4)

    axes_slider_one = pylab.axes([0.07, 0.25, 0.85, 0.04])
    slider_one = Slider(axes_slider_one,
                        label=one_lable, valmin=one_min, valmax=one_max, valinit=(one_max + one_min) / 2,
                        valfmt='%1.1f')
    slider_one.on_changed(onChangeValue)

    axes_slider_two = pylab.axes([0.07, 0.17, 0.85, 0.04])
    slider_two = Slider(axes_slider_two,
                        label=two_lable, valmin=two_min, valmax=two_max, valinit=(two_max + two_min) / 2,
                        valfmt='%1.1f')
    slider_two.on_changed(onChangeValue)

    updateGraph(slider_one.val, slider_two.val, graph_axes)
    pylab.show()


# RFR
def grid_search_rfr(xvar, yvar, n_estimators, max_depth, min_samples_leaf, min_samples_split, cv, n_iter):
    if yvar.shape[1] == 1:
        yvar = yvar.values.ravel()

    n_estimators = list(map(int, n_estimators.split(',')))
    max_depth = list(map(int, max_depth.split(',')))
    min_samples_leaf = list(map(int, min_samples_leaf.split(',')))
    min_samples_split = list(map(int, min_samples_split.split(',')))
    n_iter = int(n_iter)
    cv = int(cv)

    parameters = {'n_estimators': n_estimators, 'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}
    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)
    rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
    random_cv = RandomizedSearchCV(estimator=rfr,
                                   param_distributions=parameters,
                                   cv=ss, n_iter=n_iter,
                                   scoring='r2',
                                   n_jobs=-1,
                                   random_state=42)
    random_cv.fit(xvar, yvar)
    return list(random_cv.best_params_.items())


def graph_val_rfr_nest(xvar, yvar, cv, n_iter, n_estimators, max_depth, min_samples_leaf, min_samples_split):
    if yvar.shape[1] == 1:
        yvar = yvar.values.ravel()

    cv, n_iter = int(cv), int(n_iter)
    n_estimators = list(map(int, n_estimators.split(',')))

    parameters = {'n_estimators': n_estimators}

    rfr = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)
    random_cv = GridSearchCV(estimator=rfr, param_grid=parameters, cv=ss,
                                   scoring='r2', n_jobs=-1, return_train_score=True)
    random_cv.fit(xvar, yvar)

    cv_res = DataFrame(random_cv.cv_results_)
    cv_res = cv_res.rename(columns={'param_n_estimators': 'n_estimators',
                                    'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})

    tabl = cv_res[['n_estimators', 'mse_te', 'std_test_score', 'mse_tr', 'std_train_score']]
    tabl = tabl.assign(mean_test_score=cv_res.mse_te)
    tabl = tabl.assign(mean_train_score=cv_res.mse_tr)
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
    tabl.plot(x='n_estimators', y='mean_test_score', ax=axes[0][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='n_estimators', y='std_test_score', ax=axes[0][1]).grid(linewidth=0.2)
    tabl.plot(x='n_estimators', y='mean_train_score', ax=axes[1][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='n_estimators', y='std_train_score', ax=axes[1][1]).grid(linewidth=0.2)
    return plt.show()


def graph_val_rfr_maxd(xvar, yvar, cv, n_iter, n_estimators, max_depth, min_samples_leaf, min_samples_split):
    if yvar.shape[1] == 1:
        yvar = yvar.values.ravel()

    cv, n_iter = int(cv), int(n_iter)

    max_depth = list(map(int, max_depth.split(',')))
    parameters = {'max_depth': max_depth}

    rfr = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=n_estimators,
                                min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)

    random_cv = GridSearchCV(estimator=rfr, param_grid=parameters, cv=ss,
                             scoring='r2', n_jobs=-1, return_train_score=True)
    # random_cv = RandomizedSearchCV(estimator=rfr, param_distributions=parameters, cv=cv, n_iter=n_iter,
    #                                scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=True,
    #                                random_state=42)

    random_cv.fit(xvar, yvar)
    cv_res = DataFrame(random_cv.cv_results_)
    cv_res = cv_res.rename(columns={'param_max_depth': 'max_depth',
                                    'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})

    tabl = cv_res[['max_depth', 'mse_te', 'std_test_score', 'mse_tr', 'std_train_score']]
    tabl = tabl.assign(mean_test_score=cv_res.mse_te)
    tabl = tabl.assign(mean_train_score=cv_res.mse_tr)
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
    tabl.plot(x='max_depth', y='mean_test_score', ax=axes[0][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='max_depth', y='std_test_score', ax=axes[0][1]).grid(linewidth=0.2)
    tabl.plot(x='max_depth', y='mean_train_score', ax=axes[1][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='max_depth', y='std_train_score', ax=axes[1][1]).grid(linewidth=0.2)
    return plt.show()


def rfr_model(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, criterion):
    if max_depth != 'None':
        max_depth = int(max_depth)
    if max_depth == 'None':
        max_depth = None
    if max_features != 'auto':
        max_features = int(max_features)

    n_estimators, min_samples_leaf, min_samples_split, = int(n_estimators), int(min_samples_leaf), \
                                                         int(min_samples_split)

    reg = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=max_depth, n_estimators=n_estimators,
                                max_features=max_features, min_samples_leaf=min_samples_leaf,
                                criterion=criterion, min_samples_split=min_samples_split)
    return reg


def test_train_cv_rfr(xvar, yvar, model):
    if yvar.shape[1] == 1:
        yvar = yvar.values.ravel()
    b, b1, tmp, dt = [], [], {}, DataFrame()
    model1 = model
    for i in range(10):
        xn, xtest, yn, ytest = train_test_split(xvar, yvar)
        model1.fit(xn, yn)
        b.append(r2_score(ytest, model1.predict(xtest)))
        b1.append(mean_squared_error(ytest, model1.predict(xtest)))
    cv_r2, cv_mse = sum(b) / len(b), sum(b1) / len(b1)
    model.fit(xvar, yvar)
    tmp[''] = 'Train'
    tmp['MSE'] = mean_squared_error(yvar, model.predict(xvar))
    tmp['R2'] = r2_score(yvar, model.predict(xvar))
    dt = dt.append([tmp])
    tmp[''] = 'CV'
    tmp['MSE'] = cv_mse
    tmp['R2'] = cv_r2
    dt = dt.append([tmp])
    dt.set_index('', inplace=True)
    return dt


# GBR

def grid_search_gbr(xvar, yvar, n_estimators, max_depth, min_samples_leaf, min_samples_split, cv, n_iter):
    n_estimators = list(map(int, n_estimators.split(',')))
    max_depth = list(map(int, max_depth.split(',')))
    min_samples_leaf = list(map(int, min_samples_leaf.split(',')))
    min_samples_split = list(map(int, min_samples_split.split(',')))
    n_iter = int(n_iter)
    cv = int(cv)

    if yvar.shape[1] == 1:
        yvar_ravel = yvar.values.ravel()
        parameters = {'n_estimators': n_estimators, 'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}
        gbr = GradientBoostingRegressor(random_state=42)

    if yvar.shape[1] != 1:
        yvar_ravel = yvar
        parameters = {'estimator__n_estimators': n_estimators, 'estimator__max_depth': max_depth,
                      'estimator__min_samples_leaf': min_samples_leaf,
                      'estimator__min_samples_split': min_samples_split}
        gbr = MultiOutputRegressor(GradientBoostingRegressor(random_state=42), n_jobs=-1)

    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)
    random_cv = RandomizedSearchCV(estimator=gbr, param_distributions=parameters,
                                   cv=ss, n_iter=n_iter, scoring='neg_mean_squared_error',
                                   n_jobs=-1, random_state=42)
    random_cv.fit(xvar, yvar_ravel)

    return list(random_cv.best_params_.items())


def gbr_model(yvar, n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, loss):
    if max_features != 'auto':
        max_features = int(max_features)
    n_estimators, min_samples_leaf, min_samples_split, max_depth = \
        int(n_estimators), int(min_samples_leaf), int(min_samples_split), int(max_depth)

    reg = GradientBoostingRegressor(random_state=42, max_depth=max_depth, n_estimators=n_estimators,
                                    max_features=max_features, min_samples_leaf=min_samples_leaf,
                                    loss=loss, min_samples_split=min_samples_split)
    if yvar.shape[1] == 1:
        reg_trans = reg
    if yvar.shape[1] != 1:
        reg_trans = MultiOutputRegressor(reg, n_jobs=-1)
    return reg_trans


def graph_val_gbr_nest(xvar, yvar, cv, n_iter, n_estimators, max_depth, min_samples_leaf, min_samples_split):
    cv, n_iter = int(cv), int(n_iter)
    n_estimators = list(map(int, n_estimators.split(',')))
    if yvar.shape[1] == 1:
        yvar_tranf = yvar.values.ravel()
        parameters = {'n_estimators': n_estimators}
        gbr = GradientBoostingRegressor(random_state=42, max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

    if yvar.shape[1] != 1:
        yvar_tranf = yvar
        parameters = {'estimator__n_estimators': n_estimators}
        gbr = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split),
            n_jobs=-1)

    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)

    random_cv = GridSearchCV(estimator=gbr, param_grid=parameters, cv=ss,
                             scoring='r2', n_jobs=-1, return_train_score=True)
    # random_cv = RandomizedSearchCV(estimator=gbr, param_distributions=parameters, cv=cv, n_iter=n_iter,
    #                                scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=True,
    #                                random_state=42)

    random_cv.fit(xvar, yvar_tranf)

    cv_res = DataFrame(random_cv.cv_results_)
    if yvar.shape[1] == 1:
        cv_res = cv_res.rename(columns={'param_n_estimators': 'n_estimators',
                                        'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})
    if yvar.shape[1] != 1:
        cv_res = cv_res.rename(columns={'param_estimator__n_estimators': 'n_estimators',
                                        'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})

    tabl = cv_res[['n_estimators', 'mse_te', 'std_test_score', 'mse_tr', 'std_train_score']]
    tabl = tabl.assign(mean_test_score=cv_res.mse_te)
    tabl = tabl.assign(mean_train_score=cv_res.mse_tr)
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
    tabl.plot(x='n_estimators', y='mean_test_score', ax=axes[0][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='n_estimators', y='std_test_score', ax=axes[0][1]).grid(linewidth=0.2)
    tabl.plot(x='n_estimators', y='mean_train_score', ax=axes[1][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='n_estimators', y='std_train_score', ax=axes[1][1]).grid(linewidth=0.2)
    return plt.show()


def graph_val_gbr_maxd(xvar, yvar, cv, n_iter, n_estimators, max_depth, min_samples_leaf, min_samples_split):
    cv, n_iter = int(cv), int(n_iter)
    max_depth = list(map(int, max_depth.split(',')))
    if yvar.shape[1] == 1:
        yvar_tranf = yvar.values.ravel()
        parameters = {'max_depth': max_depth}
        gbr = GradientBoostingRegressor(random_state=42, n_estimators=n_estimators,
                                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    if yvar.shape[1] != 1:
        yvar_tranf = yvar
        parameters = {'estimator__max_depth': max_depth}
        gbr = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, n_estimators=n_estimators,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split),
            n_jobs=-1)
    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)

    random_cv = GridSearchCV(estimator=gbr, param_grid=parameters, cv=ss,
                             scoring='r2', n_jobs=-1, return_train_score=True)
    # random_cv = RandomizedSearchCV(estimator=gbr, param_distributions=parameters, cv=cv, n_iter=n_iter,
    #                                scoring='neg_mean_squared_error', n_jobs=-1, return_train_score=True,
    #                                random_state=42)

    random_cv.fit(xvar, yvar_tranf)

    cv_res = DataFrame(random_cv.cv_results_)
    if yvar.shape[1] == 1:
        cv_res = cv_res.rename(columns={'param_max_depth': 'max_depth',
                                        'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})
    if yvar.shape[1] != 1:
        cv_res = cv_res.rename(columns={'param_estimator__max_depth': 'max_depth',
                                        'mean_test_score': 'mse_te', 'mean_train_score': 'mse_tr'})

    tabl = cv_res[['max_depth', 'mse_te', 'std_test_score', 'mse_tr', 'std_train_score']]
    tabl = tabl.assign(mean_test_score=cv_res.mse_te)
    tabl = tabl.assign(mean_train_score=cv_res.mse_tr)
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 5))
    tabl.plot(x='max_depth', y='mean_test_score', ax=axes[0][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='max_depth', y='std_test_score', ax=axes[0][1]).grid(linewidth=0.2)
    tabl.plot(x='max_depth', y='mean_train_score', ax=axes[1][0], color='red').grid(linewidth=0.2)
    tabl.plot(x='max_depth', y='std_train_score', ax=axes[1][1]).grid(linewidth=0.2)
    return plt.show()
