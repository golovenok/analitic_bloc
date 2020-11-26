#!/usr/bin/python
from tkinter import *
from tkinter import ttk
from pandas import read_csv
import data.module as m
import numpy as np

DATA = 'data/dataset.csv'


class Main(Frame):
    counter = 0

    def __init__(self, root):
        super().__init__(root)
        self.init_main()

    def init_main(self):
        ttk.Button(text='Помощь').place(x=3, y=0)
        ttk.Label(text="Информация:").place(x=3, y=25)
        self.text_ds = Text(width=60, height=6)
        self.text_ds.place(x=3, y=45)
        self.text_ds.insert(1.0, read_csv(DATA, header=0,
                                          names=['Т.св.', 'Т.ф.', 'Скор.', 'Расс.', 'Глубина', 'Ширина']))
        self.scroll = Scrollbar(command=self.text_ds.yview)
        self.text_ds.config(yscrollcommand=self.scroll.set)
        self.scroll.place(x=480, y=45, height=100)

        ttk.Button(text="Таблица данных", command=self.table1, compound=TOP).place(x=3, y=150)
        ttk.Button(text="Добавить эксперимент", command=self.add_experiment, compound=TOP).place(x=110, y=150)
        ttk.Label(text="Прогнозирование", font=15).place(x=170, y=180)

        but_1 = ttk.Button(text="Обучить модель", command=self.predict_model, compound=TOP)
        but_1.place(x=3, y=180 + 25)

        self.text_predict = Entry(textvariable=StringVar(), width=81)
        self.text_predict.place(x=3, y=232)
        self.text_predict.insert(0, 'Модель не обучена. Обучение может занять несколько минут.')

    class Table(Frame):
        def __init__(self, parent=None, headings=tuple(), rows=tuple()):
            super().__init__(parent)

            table = ttk.Treeview(self, show="headings", selectmode="browse", height=25)
            table["columns"] = headings
            table["displaycolumns"] = headings


            for head in headings:
                table.heading(head, text=head, anchor=CENTER)
                table.column(head, anchor=CENTER, width=70)

            for row in rows:
                table.insert('', END, values=tuple(row))

            scrolltable = Scrollbar(self, command=table.yview)
            table.configure(yscrollcommand=scrolltable.set)
            scrolltable.pack(side=RIGHT, fill=Y)
            table.pack(expand=YES, fill=BOTH)

    def table1(self):
        self.counter += 1
        self.table = Toplevel(self)

        self.table.title('Таблица данных')

        table = self.Table(self.table,
            headings=('№','Ток сварки', 'Ток фокус.', 'Скор. сварки', 'Расстояние', 'Глубина', 'Ширина'),
            rows=(np.column_stack((np.array([str(i) for i in range(len(read_csv(DATA)))]), read_csv(DATA).values))))


        table.pack(expand=YES, fill=BOTH)

    def add_experiment(self):
        self.counter += 2
        self.add_exp = Toplevel(self)
        self.add_exp.title('Добавить эксперимент')
        self.add_exp.geometry('200x290')

        ttk.Label(self.add_exp, text='Параметры:').place(x=40+10, y=0)

        add_iw = Entry(self.add_exp, textvariable=StringVar(), width=10)
        ttk.Label(self.add_exp, text='Ток сварки:').place(x=1+10, y=25)
        add_iw.place(x=80+10, y=25)

        add_if = Entry(self.add_exp, textvariable=StringVar(), width=10)
        ttk.Label(self.add_exp, text='Ток фокус.:').place(x=1+10, y=50)
        add_if.place(x=80+10, y=50)

        add_vw = Entry(self.add_exp, textvariable=StringVar(), width=10)
        ttk.Label(self.add_exp, text='Скор. сварки:').place(x=1+10, y=75)
        add_vw.place(x=80+10, y=75)

        add_fp = Entry(self.add_exp, textvariable=StringVar(), width=10)
        ttk.Label(self.add_exp, text='Расстояние:').place(x=1+10, y=100)
        add_fp.place(x=80+10, y=100)

        ttk.Label(self.add_exp, text='Размеры шва:').place(x=40+10, y=125)

        add_depth = Entry(self.add_exp, textvariable=StringVar(), width=10)
        ttk.Label(self.add_exp, text='Глубина:').place(x=1+10, y=150)
        add_depth.place(x=80+10, y=150)

        add_width = Entry(self.add_exp, textvariable=StringVar(), width=10)
        ttk.Label(self.add_exp, text='Ширина:').place(x=1+10, y=175)
        add_width.place(x=80+10, y=175)

        def button_add():
            f = open(DATA, 'a')
            l = str(f'"{float(add_iw.get())}","{float(add_if.get())}","{float(add_vw.get())}","{float(add_fp.get())}"'
                        f',"{float(add_depth.get())}","{float(add_width.get())}"')
            f.write('\n' + l)
            f.close()
            self.text_ds.delete(1.0, END)
            self.text_ds.insert(1.0, read_csv(DATA, header=0,
                names=['Т.св.', 'Т.ф.', 'Скор.', 'Расс.', 'Глубина', 'Ширина']))

        ttk.Button(self.add_exp, text="Добавить", command=button_add, compound=TOP).place(x=50, y=200)

        def button_del():
            f = open(DATA).readlines()
            f.pop(-1)
            with open(DATA, 'w') as F:
                F.writelines(f)
            self.text_ds.delete(1.0, END)
            self.text_ds.insert(1.0, read_csv(DATA, header=0,
                names=['Т.св.', 'Т.ф.', 'Скор.', 'Расс.', 'Глубина', 'Ширина']))

        ttk.Button(self.add_exp, text="Удалить пос. эксперимент", command=button_del, compound=TOP).place(x=1+10, y=255)

    def predict_model(self):

        xvar = read_csv(DATA)[["IW", "IF", "VW", "FP"]].values
        yvar = read_csv(DATA)[["Depth", "Width"]].values

        alphas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10]
        degrees = [1, 2, 3, 4, 5]

        best_param = m.grid_search_ridge_alpha(xvar=xvar, yvar=yvar, alphas=alphas, degrees=degrees)
        model_fit = m.ridge_model(alpha=best_param[1], degree=best_param[0]).fit(xvar,yvar)
        test_score = m.test_cv(xvar=xvar, yvar=yvar, model=model_fit)

        self.text_predict.delete(0, END)
        self.text_predict.insert(0, f'Модель обучена. Качество предсказания - {round(test_score,3)*100} %.')

        Label(text='График данных', font=15).place(x=170, y=232+25)

        l = Label(text='Выберете параметр:')
        l.place(x=3, y=232+25+25+10)

        roll = ['Ток с.', 'Ток ф.', 'Скор.', 'Расс.']
        self.axis = ttk.Combobox(width=9, values=roll)
        self.axis.current(0)
        self.axis.place(x=80+3+40, y=232+25+25+10)

        def graph_data():
            self.graph_data_cust(model_fit, self.axis.get())

        ttk.Button(text='Построить график', command=graph_data, compound=TOP).place(x=3+200, y=280+10)

        Label(text='Предсказание модели', font=15).place(x=150, y=280+40)

        Label( text='Ток сварки').place(x=3, y=347)
        x1 = Entry(textvariable=StringVar(), width=7)
        x1.place(x=10, y=370)
        Label(text='Ток фокус.').place(x=3+75, y=347)
        x2 = Entry(textvariable=StringVar(), width=7)
        x2.place(x=10 + 75, y=370)
        Label(text='Скорость').place(x=3+75*2, y=347)
        x3 = Entry(textvariable=StringVar(), width=7)
        x3.place(x=10 + 75 * 2, y=370)
        Label(text='Расстояние').place(x=3+75*3, y=347)
        x4 = Entry(textvariable=StringVar(), width=7)
        x4.place(x=10 + 75 * 3, y=370)

        def predict_model():
            var_get = [float(x1.get()), float(x2.get()), float(x3.get()), float(x4.get())]
            rezult = m.r_model_predict(var_get, model=model_fit)
            t = Text(width=47, height=1)
            t.place(x=3, y=370+53)
            t.insert(1.0, f'Глубина: {round(rezult[0][0],3)}, Ширина: {round(rezult[0][1],3)}')

        ttk.Button(text='Предсказать', command=predict_model, compound=TOP).place(x=3, y=370+25)

    def graph_data_cust(self, model, x):
        dataset = read_csv(DATA)
        roll_rus = ['Ток с.', 'Ток ф.', 'Скор.', 'Расс.']
        dtx = ["IW", "IF", "VW", "FP"]
        dty = ["Глубина", "Ширина"]
        x_lable = ''
        for i in range(4):
            if x == roll_rus[i]:
                x_lable = dtx[i]

        dt1 = dtx.copy()
        dt1.remove(x_lable)
        roll_rus.remove(x)
        x_min = dataset[x_lable].min()
        x_max = dataset[x_lable].max()
        flag = dtx.index(x_lable)
        one_lable = roll_rus[0]
        one_min = dataset[dt1[0]].min()
        one_max = dataset[dt1[0]].max()
        two_lable = roll_rus[1]
        two_min = dataset[dt1[1]].min()
        two_max = dataset[dt1[1]].max()
        three_lable = roll_rus[2]
        three_min = dataset[dt1[2]].min()
        three_max = dataset[dt1[2]].max()

        m.graph_data_4var(x_lable=x, x_min=x_min, x_max=x_max, flag=flag, model=model, y_var=dty,
                          one_lable=one_lable, one_min=one_min, one_max=one_max,
                          two_lable=two_lable, two_min=two_min, two_max=two_max,
                          three_lable=three_lable, three_min=three_min, three_max=three_max)

# чч
if __name__ == "__main__":
    root = Tk()
    app = Main(root)
    app.pack()
    root.title("Прогнозирование процессов ЭЛС")
    root.geometry("500x450")
    root.mainloop()