from tkinter import *
from tkinter.filedialog import askopenfilename
import csv
import os
from tkinter import ttk
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

gui = Tk()

gui.title('Machine Learning GUI')

gui.geometry('600x600')

progress_bar = ttk.Progressbar(orient = 'horizontal', length=600, mode='determinate')
progress_bar.grid(row=150, columnspan=3, pady =10)


def data():
    global filename
    filename = askopenfilename(initialdir='C:\\',title = "Select file")
    e1.delete(0, END)
    e1.insert(0, filename)
    #e1.config(text=filename)
    #print(filename)



    import pandas as pd
    global file1

    file1 = pd.read_excel(filename)

    global col
    col = list(file1.head(0))
    #print(col)

    for i in range(len(col)):
        box1.insert(i+1, col[i])

def X_values():

    values = [box1.get(idx) for idx in box1.curselection()]
    for i in range(len(list(values))):
        box2.insert(i+1, values[i])
        box1.selection_clear(i+1, END)
    X_values.x1=[]
    for j in range(len(values)):X_values.x1.append(j)

    global x_size
    x_size = len(X_values.x1)
    print(x_size)


    print(X_values.x1)



def y_values():
    values= [box1.get(idx) for idx in box1.curselection()]
    for i in range(len(list(values))):
        box3.insert(i+1, values[i])
    y_values.y1=[]
    # for j in range(len(values)):y_values.y1.append(j)
    y_values.y1.append(9)

    print(y_values.y1)


def clear():
    pass

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (1./(2.*m)) * np.sum(np.square(AL - Y))
    
    return cost

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = {}
    L = len(layers_dims) - 1  
    m=9
    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * learning_rate #(1, 32)
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))  #0


    for i in range(num_iterations):

        # Forward 
        AL = X
        caches = []
        for l in range(1, L):
            AL_prev = AL
            Wl = parameters['W' + str(l)]
            bl = parameters['b' + str(l)]
            Zl = np.dot(Wl, AL_prev) + bl
            AL, cache = relu(Zl)
            caches.append(cache)


        # Output layer (linear activation)
        WL = parameters['W' + str(L)]
        bL = parameters['b' + str(L)]
        ZL = np.dot(WL, AL) + bL

        cost = compute_cost(ZL, Y)

        # Backward 
        dZL = (1./m) * (ZL - Y)
        grads = {}
        grads['dW' + str(L)] = np.dot(dZL, AL.T)
        grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True)

        for l in reversed(range(1, L)):
            cache = caches[l-1]
            dA = np.dot(parameters['W' + str(l+1)].T, dZL)
            dZ = relu_backward(dA, cache)
            grads['dW' + str(l)] = np.dot(dZ, AL_prev.T)
            grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True)

        # Update parameters
        for l in range(1, L+1):
            parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
            
            
    
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(costs, label='Test')     
    return ZL #чтобы вывести пред   


def sol():
    progress()
    global X
    global Y
    X = file1.iloc[:,X_values.x1].values
    Y = file1.iloc[:,y_values.y1].values
    min_max_scaler=MinMaxScaler()
    X_scale=min_max_scaler.fit_transform(X)
    Y = Y.reshape(-1, 1)
    Y_scale=min_max_scaler.fit_transform(Y)
    global x_train
    global y_train
    global y_pred
    x_train = np.transpose(X_scale)
    y_train = np.transpose(Y_scale)
    m=9
    layers_dims=[9, 32, 1]
    learning_rate=0.01
    num_iterations=10
    parameters = L_layer_model(x_train, y_train, layers_dims, learning_rate=0.01, num_iterations = 2000, print_cost=True)
    y_pred = parameters.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    # Вычисляем значения MSE, MAE и R2
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    # Создаем метки для вывода значений MSE, MAE и R2
    mse_label = Label(gui, text="MSE: {:.4f}".format(mse))
    mae_label = Label(gui, text="MAE: {:.4f}".format(mae))
    r2_label = Label(gui, text="R2: {:.4f}".format(r2))

    # Размещаем метки на форме
    mse_label.grid(row=35, column=1)
    mae_label.grid(row=40, column=1)
    r2_label.grid(row=45, column=1)



def progress():
    progress_bar['maximum']=100

    for i in range(101):
        time.sleep(0.01)
        progress_bar['value'] = i
        progress_bar.update()

    progress_bar['value'] = 0


l1=Label(gui, text='Select Data File')
l1.grid(row=0, column=0)
e1 = Entry(gui,text='')
e1.grid(row=0, column=1)

Button(gui,text='open', command=data).grid(row=0, column=2)

box1 = Listbox(gui,selectmode='multiple')
box1.grid(row=10, column=0)
Button(gui, text='Clear All',command=clear).grid(row=12,column=0)

box2 = Listbox(gui)
box2.grid(row=10, column=1)
Button(gui, text='Select X', command=X_values).grid(row=12,column=1)

box3 = Listbox(gui)
box3.grid(row=10, column=2) 
Button(gui, text='Select y', command=y_values).grid(row=12,column=2)

Button(gui, text='Solution', command=sol).grid(row=20, column=1)


def show():
    global train
    global pred
    train=y_train[1:20]
    pred=y_pred[1:20]
    plt.plot(train, label='Test')
    plt.plot(pred, label='Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()


def compare():
    train=y_train[1:20]
    pred=y_pred[1:20]
    train = train.flatten()
    pred = pred.flatten()
    min_val = np.min(Y)
    max_val = np.max(Y)
    test_denorm = train * (max_val - min_val) + min_val
    pred_denorm = pred * (max_val - min_val) + min_val
    data=[test_denorm, pred_denorm]

    table = ttk.Treeview(gui)
    # Указываем заголовки колонок
    table["columns"] = ("test", "pred")
    table.column("#0", width=0, stretch=NO)
    table.column("test", width=100)
    table.column("pred", width=100)
    table.heading("#0", text="")
    table.heading("test", text="Test")
    table.heading("pred", text="Pred")

    # Добавляем данные в таблицу
    for i in range(len(data[0])):
        table.insert("", END, text="", values=(int(data[0][i]), int(data[1][i])))

    table.grid(row=50, column=1)

Button(gui, text='Show Graphic', command=show).grid(row=30, column=1)
Button(gui, text='Compare test and prediction ', command=compare).grid(row=32, column=1)
gui.mainloop()