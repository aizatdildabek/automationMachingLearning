import os
import tkinter as tk
from tkinter import *
from tkinter.font import Font
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root= Tk()
root.title('Main runner')
root.geometry('700x700')
#root.configure(fg=)


def LinReg():
    os.startfile(r"C:\Users\user\source\repos\PythonCode\algorithms\LinReg.py",operation="open")

def RandomForest():
    os.startfile(r"C:\Users\user\source\repos\PythonCode\algorithms\RandomForest.py",operation="open")

def twolayer():
    os.startfile(r"C:\Users\user\source\repos\PythonCode\algorithms\twolayer.py",operation="open")    

def fourlayer():
    pass

my_font = Font(size=20, family="Arial")
Label(root,text="Онлайн саудадағы бағаларды бақылау үшін \n web scraper әзірлеу және бағаларды \n болжау алгоритмдерін әзірлеу",font=my_font).place(x=100,y=20) 

font2 = Font(size=15, family="Arial")
Label(root,text="Бағаларды болжау алгоритмдерін таңдаңыз", font=font2).place(x=150,y=170) 

# Создаем список вариантов
options1 = ["Linear Regression", "Random Forest Regressor"]
selected_option1 = tk.StringVar(root)
selected_option1.set("Машиналық оқыту арқылы")


def option_selectedML(*args):
    if selected_option1.get() == "Linear Regression":
        LinReg()
    elif selected_option1.get() == "Random Forest Regressor":
        RandomForest()

option_menu = tk.OptionMenu(root, selected_option1, *options1)
option_menu.place(x=150,y=200)
selected_option1.trace("w", option_selectedML)  



# Создаем список вариантов
options2 = ["2 Layer Neural Network", "4 Layer Neural Network"]
selected_option2 = tk.StringVar(root)
selected_option2.set("Жасанды желі арқылы")


def option_selectedNN(*args):
    if selected_option2.get() == "2 Layer Neural Network":
        twolayer()
    elif selected_option2.get() == "4 Layer Neural Network":
        fourlayer()

option_menu = tk.OptionMenu(root, selected_option2, *options2)
option_menu.place(x=350,y=200)
selected_option2.trace("w", option_selectedNN)  


data=pd.read_excel('C:/Users/user/source/repos/PythonCode/algorithms/merge.xlsx')
df = pd.DataFrame(data)

fig = Figure(figsize=(6, 4), dpi=90)
ax = fig.add_subplot(111)
df.set_index('Name').plot(kind='bar', ax=ax)
ax.set_xlabel('Товар')
ax.set_ylabel('Цена')
ax.set_title('Сравнение цен товара')
ax.legend(loc='upper right')


canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().place(x=100, y=300)
root.mainloop()