#!/usr/bin/python
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from pandas import read_csv
import data.module as m


class Main(Frame):
    counter = 0

    def __init__(self, root):
        super().__init__(root)
        self.init_main()

    def init_main(self):

        btn_open_file = ttk.Button(text='Open', command=self.open_file, compound=TOP)
        btn_open_file.place(x=0, y=0)

    def open_file(self):
        self.fname = askopenfilename(filetypes=(("Comma Separated Values", "*.csv"), ("All files", "*.*")))
        self.dt = read_csv(self.fname)
        self.text0 = ttk.Label(text="Dataset:")
        self.text = Text(width=100, height=4)
        self.text0.place(x=0, y=24)
        self.text.place(x=0, y=42)
        self.text.insert(1.0, self.dt[:])

        btn_1 = ttk.Button(text='Ridge', command=self.Ridge, compound=TOP)
        btn_1.place(x=75, y=0)

        btn_2 = ttk.Button(text='RFR', command=self.Random_Forest, compound=TOP)
        btn_2.place(x=150, y=0)

        btn_2_1 = ttk.Button(text='GBR', command=self.Gradient_Boosting, compound=TOP)
        btn_2_1.place(x=150 + 75, y=0)

        l1 = ttk.Label(text="Choose values X:")
        l1.place(x=0, y=115)
        self.x_var = {}
        for i in range(len(self.dt.columns.values)):
            x = BooleanVar()
            x.set(0)
            cx = ttk.Checkbutton(text=self.dt.columns.values[i], variable=x, onvalue=1, offvalue=0)
            cx.place(x=i * 70, y=140)
            self.x_var[self.dt.columns.values[i]] = x

        l2 = Label(text="Choose values Y:")
        l2.place(x=0, y=165)
        self.y_var = {}
        for i in range(len(self.dt.columns.values)):
            y = BooleanVar()
            y.set(0)
            cx = ttk.Checkbutton(text=self.dt.columns.values[i], variable=y, onvalue=1, offvalue=0)
            cx.place(x=i * 70, y=190)
            self.y_var[self.dt.columns.values[i]] = y

        btn_3 = ttk.Button(text='Corr matrix', command=self.corr_matrix, compound=TOP)
        btn_3.place(x=80, y=220)
        btn_4 = ttk.Button(text='Describe', command=self.describe, compound=TOP)
        btn_4.place(x=0, y=220)

    def xdt(self):
        self.x_all = []
        for i in range(len(self.x_var)):
            b = list(self.x_var.items())
            if b[i][1].get() == 1:
                self.x_all.append(b[i][0])
        return self.dt[self.x_all]

    def ydt(self):
        self.y_all = []
        for i in range(len(self.y_var)):
            b = list(self.y_var.items())
            if b[i][1].get() == 1:
                self.y_all.append(b[i][0])
        return self.dt[self.y_all]

    def corr_matrix(self):
        self.xdt()
        self.ydt()
        dt = self.dt[self.x_all + self.y_all]
        m.corr_matrix(dt)

    def describe(self):
        self.xdt()
        self.ydt()
        dt = self.dt[self.x_all + self.y_all]
        text1 = Text(width=100, height=9)
        text1.insert(1.0, m.describe(dt))
        text1.place(x=0, y=250)

    def Ridge(self):
        self.counter += 1
        self.rid = Toplevel(self)
        self.rid.title('Ridge Regressor')
        self.rid.geometry('400x500')
        btn_5 = ttk.Button(self.rid, text='Help')
        btn_5.place(x=0, y=0)
        l = Label(self.rid, text='Grid Search:', font=15)
        sm = 30
        l.place(x=0, y=sm)
        l1 = Label(self.rid, text='(cv =        )')
        l1.place(x=120, y=sm + 4)

        self.cv_test_entry = Entry(self.rid, textvariable=StringVar(), width=3)
        self.cv_test_entry.place(x=120 + 30, y=35)
        self.cv_test_entry.insert(0, '5')

        l = Label(self.rid, text='Enter options for search: ')
        l.place(x=0, y=20 + sm + 3)

        l = Label(self.rid, text='Degrees: ')
        l.place(x=0, y=45 + sm)
        self.degree_entry = Entry(self.rid, textvariable=StringVar(), width=50)
        self.degree_entry.place(x=60, y=45 + sm)
        self.degree_entry.insert(0, "1, 2, 3, 4, 5")

        l = Label(self.rid, text='Alphas: ')
        l.place(x=0, y=65 + sm)
        self.alpha_entry = Entry(self.rid, textvariable=StringVar(), width=50)
        self.alpha_entry.place(x=60, y=70 + sm)
        self.alpha_entry.insert(0, "1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1")

        btn_6 = ttk.Button(self.rid, text='Run', command=self.grid_search_ridge, compound=TOP)
        btn_6.place(x=0, y=95 + sm)

        l = Label(self.rid, text='Fit model', font=15)
        l.place(x=0, y=173)

        l = Label(self.rid, text='Degree: ')
        l.place(x=0, y=190 + 8)
        self.degree_model = Entry(self.rid, textvariable=StringVar(), width=6)
        self.degree_model.place(x=50, y=190 + 10)

        l = Label(self.rid, text='Alpha: ')
        l.place(x=0, y=219)
        self.alpha_model = Entry(self.rid, textvariable=StringVar(), width=6)
        self.alpha_model.place(x=50, y=190 + 32)

        btn_7 = ttk.Button(self.rid, text='Fit', command=self.model_ridge, compound=TOP)
        btn_7.place(x=0, y=245)

        self.xdt()
        l = Label(self.rid, text='Model prediction', font=15)
        l.place(x=0, y=310 + 15)
        var_predict = []
        n = 0
        for i in self.x_all:
            l = Label(self.rid, text=i)
            l.place(x=n * 70, y=330 + 20)
            x = StringVar()
            x_entry = Entry(self.rid, textvariable=x, width=5)
            x_entry.place(x=n * 70, y=350 + 20)
            n += 1
            var_predict.append(x_entry)

        def predict_model():
            var_get = [float(i.get()) for i in var_predict]
            rezult = m.r_model_pridict(var_get, model=self.r_model_fit)
            t = Text(self.rid, width=45, height=1)
            t.place(x=0, y=415 + 5)
            j = ', '.join(self.y_all)
            t.insert(1.0, f'{j}: {rezult[0]}')

        self.btn_7 = ttk.Button(self.rid, text='Predict', command=predict_model, compound=TOP)
        self.btn_7.place(x=0, y=385 + 10)

    def model_ridge(self):
        r_model = m.ridge_model(self.alpha_model.get(), self.degree_model.get())
        test_train_cv = m.test_train_cv(self.xdt(), self.ydt(), r_model)
        text = Text(self.rid, width=35, height=4)
        text.insert(1.0, test_train_cv)
        text.place(x=95, y=173 + 20)

        self.r_model_fit = r_model.fit(self.xdt(), self.ydt())

        # graph
        l = Label(self.rid, text='Data graph.', font=15)
        l.place(x=0, y=270)

        l = Label(self.rid, text='Abscissa axis: ')
        l.place(x=0, y=280 + 20)

        self.axis_r = ttk.Combobox(self.rid, width=7, values=self.x_all)
        self.axis_r.current(0)
        self.axis_r.place(x=80, y=280 + 20)

        def graph_data_ridge():
            self.graph_data(self.r_model_fit, self.axis_r.get())

        btn_8 = ttk.Button(self.rid, text='Graph Data', command=graph_data_ridge, compound=TOP)
        btn_8.place(x=150, y=280 + 18)

    def graph_data(self, model, x_lable):
        dtx = self.x_all
        dty = self.y_all

        # x_lable = self.axis.get()
        dt1 = dtx.copy()
        dt1.remove(x_lable)

        x_min = self.xdt()[x_lable].min()
        x_max = self.xdt()[x_lable].max()
        flag = dtx.index(x_lable)
# + -
        one_lable = dt1[0]
        one_min = self.xdt()[dt1[0]].min()
        one_max = self.xdt()[dt1[0]].max()

        two_lable = dt1[1]
        two_min = self.xdt()[dt1[1]].min()
        two_max = self.xdt()[dt1[1]].max()

        if len(dtx) == 3:
            m.graph_data_3var(x_lable=x_lable, x_min=x_min, x_max=x_max, flag=flag, model=model, y_var=dty,
                              one_lable=one_lable, one_min=one_min, one_max=one_max,
                              two_lable=two_lable, two_min=two_min, two_max=two_max)
        if len(dtx) == 4:
            three_lable = dt1[2]
            three_min = self.xdt()[dt1[2]].min()
            three_max = self.xdt()[dt1[2]].max()
            m.graph_data_4var(x_lable=x_lable, x_min=x_min, x_max=x_max, flag=flag, model=model, y_var=dty,
                              one_lable=one_lable, one_min=one_min, one_max=one_max,
                              two_lable=two_lable, two_min=two_min, two_max=two_max,
                              three_lable=three_lable, three_min=three_min, three_max=three_max)

    def grid_search_ridge(self):
        model = m.grid_search_ridge(
            self.xdt(), self.ydt(), self.alpha_entry.get(), self.degree_entry.get(), self.cv_test_entry.get())
        text = Text(self.rid, width=45, height=1)
        text.insert(1.0, f' Degree: {model[0]}, Alpha: {model[1]}')
        text.place(x=0, y=152)
        self.degree_model.delete(0, END)
        self.degree_model.insert(0, model[0])
        self.alpha_model.delete(0, END)
        self.alpha_model.insert(0, model[1])

        def graph_val_ridge_alpha():
            m.graph_val_ridge_alpha(xvar=self.xdt(), yvar=self.ydt(), degree=model[0],
                                    alphas=self.alpha_entry.get(), cv=self.cv_test_entry.get())

        btn_read_text = ttk.Button(self.rid, text='Graph Alpha', command=graph_val_ridge_alpha, compound=TOP)
        btn_read_text.place(x=90 + 80, y=95 + 30)

        def graph_val_ridge_degree():
            m.graph_val_ridge_degree(self.xdt(), self.ydt(), self.degree_entry.get(), model[1], cv=self.cv_test_entry.get())

        btn_read_text = ttk.Button(self.rid, text='Graph Degrees', command=graph_val_ridge_degree, compound=TOP)
        btn_read_text.place(x=80, y=95 + 30)

    def Random_Forest(self):
        self.counter += 2
        self.rfr = Toplevel(self)
        self.rfr.title('Random Forest Regressor')
        self.rfr.geometry('400x580')
        btn_10 = ttk.Button(self.rfr, text='Help')
        btn_10.place(x=0, y=0)
        sm = 30
        l = Label(self.rfr, text='Randomized Search:', font=15)
        l.place(x=0, y=sm)
        l1 = Label(self.rfr, text='(n_iter =         ; cv =        )')
        l1.place(x=175+10, y=sm+5)
        self.n_iter_entry = Entry(self.rfr, textvariable=StringVar(), width=4)
        self.n_iter_entry.place(x=180 + 50, y=32 + 5)
        self.n_iter_entry.insert(0, '100')

        self.cv_2_entry = Entry(self.rfr, textvariable=StringVar(), width=3)
        self.cv_2_entry.place(x=238 + 50, y=32 + 5)
        self.cv_2_entry.insert(0, '5')

        l = Label(self.rfr, text='Enter options for search: ')
        l.place(x=0, y=20 + sm + 2)

        l = Label(self.rfr, text='N_estimators: ')
        l.place(x=0, y=44 + sm)
        self.n_estimators_entry = Entry(self.rfr, textvariable=StringVar(), width=45)
        self.n_estimators_entry.place(x=80 + 30, y=45 + sm)
        self.n_estimators_entry.insert(1, "25, 50, 75, 100")

        l = Label(self.rfr, text='Max_depth: ')
        l.place(x=0, y=67 + sm)
        self.max_depth_entry = Entry(self.rfr, textvariable=StringVar(), width=45)
        self.max_depth_entry.place(x=80 + 30, y=70 + sm)
        self.max_depth_entry.insert(1, "2, 5, 7, 10")

        l = Label(self.rfr, text='Min_samples_leaf: ')
        l.place(x=0, y=122)
        self.min_samples_leaf_entry = Entry(self.rfr, textvariable=StringVar(), width=45)
        self.min_samples_leaf_entry.place(x=80 + 30, y=125)
        self.min_samples_leaf_entry.insert(1, "1, 2, 3")

        l = Label(self.rfr, text='Min_samples_split: ')
        l.place(x=0, y=122 + 25)
        self.min_samples_split_entry = Entry(self.rfr, textvariable=StringVar(), width=45)
        self.min_samples_split_entry.place(x=80 + 30, y=125 + 25)
        self.min_samples_split_entry.insert(1, "2, 3, 4")

        btn_11 = ttk.Button(self.rfr, text='Run', command=self.grid_search_rfr, compound=TOP)
        btn_11.place(x=0, y=172)

        l = Label(self.rfr, text='Fit model', font=15)
        l.place(x=0, y=232)

        l = Label(self.rfr, text='N_estimators: ')
        l.place(x=0, y=258)
        self.n_estimators_fit = Entry(self.rfr, textvariable=StringVar(), width=8)
        self.n_estimators_fit.place(x=90 + 20, y=260)

        l = Label(self.rfr, text='Max_depth: ')
        l.place(x=0, y=190 + 29 + 60)
        self.max_depth_fit = Entry(self.rfr, textvariable=StringVar(), width=8)
        self.max_depth_fit.place(x=90 + 20, y=190 + 32 + 60)

        l = Label(self.rfr, text='Min_samples_leaf: ')
        l.place(x=0, y=240 + 60)
        self.min_samples_leaf_fit = Entry(self.rfr, textvariable=StringVar(), width=8)
        self.min_samples_leaf_fit.place(x=90 + 20, y=242 + 60)

        l = Label(self.rfr, text='Min_samples_split: ')
        l.place(x=0, y=240 + 20 + 60)
        self.min_samples_split_fit = Entry(self.rfr, textvariable=StringVar(), width=8)
        self.min_samples_split_fit.place(x=90 + 20, y=322)

        l = Label(self.rfr, text='Max_features: ')
        l.place(x=0, y=340)
        self.max_features_fit = Entry(self.rfr, textvariable=StringVar(), width=8)
        self.max_features_fit.place(x=90 + 20, y=342)
        self.max_features_fit.insert(1, 'auto')

        l = Label(self.rfr, text='Criterion: ')
        l.place(x=0, y=360)
        self.criterion_fit = ttk.Combobox(self.rfr, width=5, values=['mse', 'mae'])
        self.criterion_fit.current(0)
        self.criterion_fit.place(x=90 + 20, y=362)

        btn_12 = ttk.Button(self.rfr, text='Fit', command=self.model_rfr, compound=TOP)
        btn_12.place(x=0, y=362 + 22)

        self.xdt()
        l = Label(self.rfr, text='Model prediction', font=15)
        l.place(x=0, y=404 + 10+37)
        var_predict = []
        n = 0
        for i in self.x_all:
            l = Label(self.rfr, text=i)
            l.place(x=n * 70, y=479)
            x_entry = Entry(self.rfr, textvariable=StringVar(), width=5)
            x_entry.place(x=n * 70, y=499)
            n += 1
            var_predict.append(x_entry)

        def predict_model():
            var_get = [float(i.get()) for i in var_predict]
            rezult = m.r_model_pridict(var_get, model=self.rfr_model_fit)
            t = Text(self.rfr, width=47, height=1)
            t.place(x=0, y=504+45)
            j = ', '.join(self.y_all)
            t.insert(1.0, f'{j}: {rezult[0]}')

        btn_18 = ttk.Button(self.rfr, text='Predict', command=predict_model, compound=TOP)
        btn_18.place(x=0, y=479+45)

    def model_rfr(self):
        rfr_model = m.rfr_model(n_estimators=self.n_estimators_fit.get(),
                                max_depth=self.max_depth_fit.get(),
                                min_samples_leaf=self.min_samples_leaf_fit.get(),
                                min_samples_split=self.min_samples_split_fit.get(),
                                max_features=self.max_features_fit.get(),
                                criterion=self.criterion_fit.get())

        test_train_cv = m.test_train_cv_rfr(self.xdt(), self.ydt(), rfr_model)
        roun = [round(i, 2) for i in rfr_model.feature_importances_]
        text = Text(self.rfr, width=27, height=8)
        text.insert(1.0, f'{test_train_cv}\n\nFeature importances:\n{self.x_all}\n{roun}')
        text.place(x=170, y=261)

        if self.ydt().shape[1] == 1:
            yvar = self.ydt().values.ravel()
        if self.ydt().shape[1] != 1:
            yvar = self.ydt()

        self.rfr_model_fit = rfr_model.fit(self.xdt(), yvar)

        l = Label(self.rfr, text='Data graph', font=13)
        l.place(x=0, y=407)

        l = Label(self.rfr, text='Abscissa axis: ')
        l.place(x=0, y=410+25)

        self.axis_rfr = ttk.Combobox(self.rfr, width=7, values=self.x_all)
        self.axis_rfr.current(0)
        self.axis_rfr.place(x=80, y=410+25)

        def graph_data_rfr():
            self.graph_data(self.rfr_model_fit, self.axis_rfr.get())

        btn_191 = ttk.Button(self.rfr, text='Graph Data', command=graph_data_rfr, compound=TOP)
        btn_191.place(x=150, y=410+23)

    def grid_search_rfr(self):
        model = m.grid_search_rfr(xvar=self.xdt(), yvar=self.ydt(), n_estimators=self.n_estimators_entry.get(),
                                  max_depth=self.max_depth_entry.get(),
                                  min_samples_leaf=self.min_samples_leaf_entry.get(),
                                  min_samples_split=self.min_samples_split_entry.get(),
                                  cv=self.cv_2_entry.get(), n_iter=self.n_iter_entry.get())

        text = Text(self.rfr, width=47, height=2)
        text.insert(1.0, f'N_estimators: {model[0][1]}, Max_depth: {model[3][1]}\n'
                         f'Min_samples_leaf: {model[2][1]}, Min_samples_split: {model[1][1]}')
        text.place(x=0, y=172 + 20 + 5)

        self.n_estimators_fit.delete(0, END)
        self.n_estimators_fit.insert(0, model[0][1])
        self.max_depth_fit.delete(0, END)
        self.max_depth_fit.insert(0, model[3][1])
        self.min_samples_leaf_fit.delete(0, END)
        self.min_samples_leaf_fit.insert(0, model[2][1])
        self.min_samples_split_fit.delete(0, END)
        self.min_samples_split_fit.insert(0, model[1][1])

        def graph_val_rfr_nest():
            m.graph_val_rfr_nest(xvar=self.xdt(), yvar=self.ydt(), cv=self.cv_2_entry.get(),
                                 n_iter=self.n_iter_entry.get(), n_estimators=self.n_estimators_entry.get(),
                                 max_depth=model[3][1], min_samples_leaf=model[2][1], min_samples_split=model[1][1])

        btn_13 = ttk.Button(self.rfr, text='Graph N_estimators', command=graph_val_rfr_nest, compound=TOP)
        btn_13.place(x=80, y=172)

        def graph_val_rfr_maxd():
            m.graph_val_rfr_maxd(xvar=self.xdt(), yvar=self.ydt(), cv=self.cv_2_entry.get(),
                                 n_iter=self.n_iter_entry.get(), max_depth=self.max_depth_entry.get(),
                                 n_estimators=model[0][1], min_samples_leaf=model[2][1], min_samples_split=model[1][1])

        btn_14 = ttk.Button(self.rfr, text='Graph Max_depth', command=graph_val_rfr_maxd, compound=TOP)
        btn_14.place(x=180 + 18, y=172)

    def Gradient_Boosting(self):
        self.counter += 3
        self.gbr = Toplevel(self)
        self.gbr.title('Gradient Boosting Regressor')
        self.gbr.geometry('400x580')
        btn_10 = ttk.Button(self.gbr, text='Help')
        btn_10.place(x=0, y=0)
        sm = 30
        l = Label(self.gbr, text='Randomized Search:', font=15)
        l.place(x=0, y=sm)
        l1 = Label(self.gbr, text='(n_iter =         ; cv =        )')
        l1.place(x=175+10, y=sm+5)
        self.n_iter3_entry = Entry(self.gbr, textvariable=StringVar(), width=4)
        self.n_iter3_entry.place(x=180 + 50, y=32 + 5)
        self.n_iter3_entry.insert(0, '100')

        self.cv3_entry = Entry(self.gbr, textvariable=StringVar(), width=3)
        self.cv3_entry.place(x=238 + 50, y=32 + 5)
        self.cv3_entry.insert(0, '5')

        l = Label(self.gbr, text='Enter options for search: ')
        l.place(x=0, y=20 + sm + 2)

        l = Label(self.gbr, text='N_estimators: ')
        l.place(x=0, y=44 + sm)
        self.n_estimators3_entry = Entry(self.gbr, textvariable=StringVar(), width=45)
        self.n_estimators3_entry.place(x=80 + 30, y=45 + sm)
        self.n_estimators3_entry.insert(1, "20, 40, 60, 80, 100")

        l = Label(self.gbr, text='Max_depth: ')
        l.place(x=0, y=67 + sm)
        self.max_depth3_entry = Entry(self.gbr, textvariable=StringVar(), width=45)
        self.max_depth3_entry.place(x=80 + 30, y=70 + sm)
        self.max_depth3_entry.insert(1, "1, 2, 3, 4, 5")

        l = Label(self.gbr, text='Min_samples_leaf: ')
        l.place(x=0, y=122)
        self.min_samples_leaf3_entry = Entry(self.gbr, textvariable=StringVar(), width=45)
        self.min_samples_leaf3_entry.place(x=80 + 30, y=125)
        self.min_samples_leaf3_entry.insert(1, "1, 2, 3, 4")

        l = Label(self.gbr, text='Min_samples_split: ')
        l.place(x=0, y=122 + 25)
        self.min_samples_split3_entry = Entry(self.gbr, textvariable=StringVar(), width=45)
        self.min_samples_split3_entry.place(x=80 + 30, y=125 + 25)
        self.min_samples_split3_entry.insert(1, "2, 3, 4, 5")

        btn_11 = ttk.Button(self.gbr, text='Run', command=self.grid_search_gbr, compound=TOP)
        btn_11.place(x=0, y=172)

        l = Label(self.gbr, text='Fit model', font=15)
        l.place(x=0, y=232)

        l = Label(self.gbr, text='N_estimators: ')
        l.place(x=0, y=258)
        self.n_estimators3_fit = Entry(self.gbr, textvariable=StringVar(), width=8)
        self.n_estimators3_fit.place(x=90 + 20, y=260)

        l = Label(self.gbr, text='Max_depth: ')
        l.place(x=0, y=190 + 29 + 60)
        self.max_depth3_fit = Entry(self.gbr, textvariable=StringVar(), width=8)
        self.max_depth3_fit.place(x=90 + 20, y=190 + 32 + 60)

        l = Label(self.gbr, text='Min_samples_leaf: ')
        l.place(x=0, y=240 + 60)
        self.min_samples_leaf3_fit = Entry(self.gbr, textvariable=StringVar(), width=8)
        self.min_samples_leaf3_fit.place(x=90 + 20, y=242 + 60)

        l = Label(self.gbr, text='Min_samples_split: ')
        l.place(x=0, y=240 + 20 + 60)
        self.min_samples_split3_fit = Entry(self.gbr, textvariable=StringVar(), width=8)
        self.min_samples_split3_fit.place(x=90 + 20, y=322)

        l = Label(self.gbr, text='Max_features: ')
        l.place(x=0, y=340)
        self.max_features3_fit = Entry(self.gbr, textvariable=StringVar(), width=8)
        self.max_features3_fit.place(x=90 + 20, y=342)
        self.max_features3_fit.insert(1, 'auto')

        l = Label(self.gbr, text='loss: ')
        l.place(x=0, y=360)
        self.loss_fit = ttk.Combobox(self.gbr, width=5, values=['ls', 'lad', 'huber'])
        self.loss_fit.current(0)
        self.loss_fit.place(x=90 + 20, y=362)

        btn_12 = ttk.Button(self.gbr, text='Fit', command=self.model_gbr, compound=TOP)
        btn_12.place(x=0, y=362 + 22)

        self.xdt()
        l = Label(self.gbr, text='Model prediction', font=15)
        l.place(x=0, y=404 + 10+45)
        var_predict = []
        n = 0
        for i in self.x_all:
            l = Label(self.gbr, text=i)
            l.place(x=n * 70, y=404 + 20 + 10+45)
            x_entry = Entry(self.gbr, textvariable=StringVar(), width=5)
            x_entry.place(x=n * 70, y=404 + 40 + 10+45)
            n += 1
            var_predict.append(x_entry)

        def predict_model():
            var_get = [float(i.get()) for i in var_predict]
            rezult = m.r_model_pridict(var_get, model=self.gbr_model_fit)
            t = Text(self.gbr, width=47, height=1)
            t.place(x=0, y=504+45)
            j = ', '.join(self.y_all)
            t.insert(1.0, f'{j}: {rezult[0]}')

        btn_18 = ttk.Button(self.gbr, text='Predict', command=predict_model, compound=TOP)
        btn_18.place(x=0, y=479+45)

    def model_gbr(self):
        gbr_model = m.gbr_model(n_estimators=self.n_estimators3_fit.get(),
                                max_depth=self.max_depth3_fit.get(),
                                min_samples_leaf=self.min_samples_leaf3_fit.get(),
                                min_samples_split=self.min_samples_split3_fit.get(),
                                max_features=self.max_features3_fit.get(),
                                loss=self.loss_fit.get(), yvar=self.ydt())

        test_train_cv = m.test_train_cv_rfr(self.xdt(), self.ydt(), gbr_model)
        text = Text(self.gbr, width=27, height=9)
        if self.ydt().shape[1] == 1:
            roun = [round(i, 2) for i in gbr_model.feature_importances_]
            text.insert(1.0, f'{test_train_cv}\n\nFeature importances:\n{self.x_all}\n{roun}')
            yvar = self.ydt().values.ravel()

        if self.ydt().shape[1] != 1:
            r1 = [round(i, 2) for i in gbr_model.estimators_[0].feature_importances_]
            r2 = [round(i, 2) for i in gbr_model.estimators_[1].feature_importances_]
            text.insert(1.0, f'{test_train_cv}\n\nFeature importances:'
                             f'\n{self.x_all}\n{self.y_all[0]}:\n{r1}\n{self.y_all[1]}:\n{r2}')
            yvar = self.ydt()

        text.place(x=170, y=261)

        self.gbr_model_fit = gbr_model.fit(self.xdt(), yvar)

        l = Label(self.gbr, text='Data graph', font=13)
        l.place(x=0, y=407)

        l = Label(self.gbr, text='Abscissa axis: ')
        l.place(x=0, y=410 + 25)

        self.axis_gbr = ttk.Combobox(self.gbr, width=7, values=self.x_all)
        self.axis_gbr.current(0)
        self.axis_gbr.place(x=80, y=410 + 25)

        def graph_data_gbr():
            self.graph_data(self.gbr_model_fit, self.axis_gbr.get())

        btn_191 = ttk.Button(self.gbr, text='Graph Data', command=graph_data_gbr, compound=TOP)
        btn_191.place(x=150, y=410 + 23)


    def grid_search_gbr(self):
        model = m.grid_search_gbr(xvar=self.xdt(), yvar=self.ydt(), n_estimators=self.n_estimators3_entry.get(),
                                  max_depth=self.max_depth3_entry.get(),
                                  min_samples_leaf=self.min_samples_leaf3_entry.get(),
                                  min_samples_split=self.min_samples_split3_entry.get(),
                                  cv=self.cv3_entry.get(), n_iter=self.n_iter3_entry.get())

        text = Text(self.gbr, width=47, height=2)
        text.insert(1.0, f'N_estimators: {model[0][1]}, Max_depth: {model[3][1]}\n'
                         f'Min_samples_leaf: {model[2][1]}, Min_samples_split: {model[1][1]}')
        text.place(x=0, y=172 + 20 + 5)

        self.n_estimators3_fit.delete(0, END)
        self.n_estimators3_fit.insert(0, model[0][1])
        self.max_depth3_fit.delete(0, END)
        self.max_depth3_fit.insert(0, model[3][1])
        self.min_samples_leaf3_fit.delete(0, END)
        self.min_samples_leaf3_fit.insert(0, model[2][1])
        self.min_samples_split3_fit.delete(0, END)
        self.min_samples_split3_fit.insert(0, model[1][1])

        def graph_val_gbr_nest():
            m.graph_val_gbr_nest(xvar=self.xdt(), yvar=self.ydt(), cv=self.cv3_entry.get(),
                                 n_iter=self.n_iter3_entry.get(), n_estimators=self.n_estimators3_entry.get(),
                                 max_depth=model[3][1], min_samples_leaf=model[2][1], min_samples_split=model[1][1])

        btn_13 = ttk.Button(self.gbr, text='Graph N_estimators', command=graph_val_gbr_nest, compound=TOP)
        btn_13.place(x=80, y=172)

        def graph_val_gbr_maxd():
            m.graph_val_gbr_maxd(xvar=self.xdt(), yvar=self.ydt(), cv=self.cv3_entry.get(),
                                 n_iter=self.n_iter3_entry.get(), max_depth=self.max_depth3_entry.get(),
                                 n_estimators=model[0][1],  min_samples_leaf=model[2][1], min_samples_split=model[1][1])

        btn_14 = ttk.Button(self.gbr, text='Graph Max_depth', command=graph_val_gbr_maxd, compound=TOP)
        btn_14.place(x=180+18, y=172)


if __name__ == "__main__":
    root = Tk()
    app = Main(root)
    app.pack()
    root.title("ЭСИМ")
    root.geometry("650x450+300+200")
    # root.resizable(False, False)
    root.mainloop()