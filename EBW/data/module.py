from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pylab
from matplotlib.widgets import Slider


def grid_search_ridge(xvar, yvar, alphas, degrees, cv=10):

    param_grid = [{'ridge__alpha': alphas, 'polynomialfeatures__degree': degrees}]
    reg_d = make_pipeline(
        PolynomialFeatures(include_bias=False),
        StandardScaler(),
        Ridge()
    )
    ss = ShuffleSplit(n_splits=cv, test_size=0.25, random_state=42)
    grid_search = GridSearchCV(reg_d, param_grid, n_jobs=-1, cv=ss, scoring='r2')

    grid_search.fit(xvar, yvar)
    return list(grid_search.best_params_.items())[0][1], list(grid_search.best_params_.items())[1][1]


def grid_search_ridge_alpha(xvar, yvar, alphas, degrees):
    best_alpha = grid_search_ridge(xvar, yvar, alphas, degrees)[1]
    alphas_2, al_1, al_2 = [], 0, best_alpha
    for i in range(9):
        al_1 += best_alpha
        al_2 -= best_alpha*0.1
        alphas_2.append(al_1)
        alphas_2.append(al_2)

    return grid_search_ridge(xvar, yvar, alphas_2, degrees)


def ridge_model(alpha, degree):
    alpha = float(alpha)
    degree = int(degree)
    reg = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha)
    )
    return reg


def test_cv(xvar, yvar, model):
    ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=10)
    return cross_val_score(model, xvar, yvar, cv=ss, scoring='r2').mean()


def r_model_predict(arg, model):
    x_matrix = np.array([arg]).reshape(1, -1)
    return model.predict(x_matrix)


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
            s.append(r_model_predict(arg, model)[0])
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

    fig.canvas.set_window_title('График данных')

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



