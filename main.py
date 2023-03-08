from numpy import random 
import numpy as np 
from time import perf_counter 
import csv 
from pandas import * 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.animation import FuncAnimation, PillowWriter 
 
 
def request_1(): 
    N_SIZE = 1000000 
    RANDOM_SEED = 100000 
 
    t1_start = perf_counter() 
    A1 = random.randint(RANDOM_SEED, size=N_SIZE) 
    B1 = random.randint(RANDOM_SEED, size=N_SIZE) 
    C1 = np.multiply(A1, B1) 
    t1_close = perf_counter() 
 
    t2_start = perf_counter() 
    A2 = [random.randint(RANDOM_SEED) for i in range(N_SIZE)] 
    B2 = [random.randint(RANDOM_SEED) for i in range(N_SIZE)] 
    C2 = [] 
    for i in range(len(A2)): 
        C2.append(A2[i]*B2[i]) 
    t2_close = perf_counter() 
    
    print(f"\t Time 1 with regular array: {t2_close - t2_start}") 
    print(f"\t Time 2 with NumPy:         {t1_close-t1_start}") 


def request_2(): 
    # Создайте точечную диаграмму в 2 столбца с соответствующими метками
    def scatter_graph(values, labels): 
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(9,7)) 
        # Создадим ax2, чтобы у меня было 2 оси Y 
        ax2 = ax1.twinx() 
 
        ax1.scatter(values[0], values[1], label=labels[1], color='cyan')
        ax2.scatter(values[0], values[2], label=labels[2], color='red')

        ax1.set_xlabel(labels[0]) 
        ax1.set_ylabel(labels[1], color='cyan') 
        ax2.set_ylabel(labels[2], color='red') 
 
        fig.legend(loc="upper right") 
        plt.title(f"График {labels[1]} и {labels[2]} от {labels[0]}") 
        plt.show() 
 
    def density_curve(values, labels): 
        # Изменение диапазона
        # Поскольку интервал не равномерный (один 0 -> 100, 0 -> 350) 
 
        min_val = min(values[2]) 
        max_val = max(values[2]) 
 
        dataset = DataFrame( 
            { 
                labels[0]: values[0], 
                labels[1]: values[1], 
                # [x, y] -> [z, t] ~ [0, y - x] -> [0, t - z] 
                # i in [x, y] -> (i - x) / (y - x) * (t - z) + z 
                labels[2]: [((i - min_val) * 100) / 
                            (max_val - min_val) for i in values[2]] 
            } 
        ) 
        
        plt.style.use('dark_background') 
        fig, ax1 = plt.subplots(figsize=(9,7)) 
 
        # Построить гистограммы и кривые плотности для столбца 4.
        sns.distplot(dataset[[labels[1]]], label=labels[1], 
                     color='cyan') 
 
        # Построить гистограмму и кривую плотности для столбца 16. 
        sns.distplot(dataset[[labels[2]]], 
                     label=labels[2], color='red') 
 
        ax1.set_ylabel("Density") 
        fig.legend(loc="upper right") 
        plt.title(f'График корреляции {labels[1]} и {labels[2]}') 
        plt.show() 
  
    columns = [] 
    line_1 = [] 
    line_4 = [] 
    line_18 = [] 
 
    # Используя приложенный файл data1.csv, подсчитать количество записей в нём 
    with open("data1.csv", 'rt', newline='', encoding="windows-1251") as csv_file: 
        data = csv.reader(csv_file, delimiter=";") 
        count = 0 
        for row in data: 
            if count == 0: 
                count = 1 
                columns = row 
            else: 
                count += 1 
                line_1.append(float(row[0])) 
                line_4.append(float(row[3])) 
                line_18.append(float(row[17])) 
 
    scatter_graph([line_1, line_4, line_18], [columns[0], columns[3], columns[17]]) 
    density_curve([line_1, line_4, line_18], [columns[0], columns[3], columns[17]]) 
 
 
def request_3(): 
    x_line = np.linspace(-5*np.pi, 5*np.pi, 100) 
    z_line = np.sin(x_line)
    y_line = np.cos(x_line)
 
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,6)) 
    ax = fig.add_subplot(111, projection='3d') 
 
    ax.plot(x_line, y_line, z_line) 
 
    ax.set_xlabel('X', color="cyan") 
    ax.set_ylabel('Y', color="cyan") 
    ax.set_zlabel('Z', color="cyan") 
 
    line, = ax.plot([], [], [], lw=5) 
 
    # 0 -> i - 1 
    # Функция, которая будет вызываться на каждом кадре анимации 
    def animate(i): 
        line.set_data(x_line[:i], y_line[:i]) 
        line.set_3d_properties(z_line[:i]) 
        return (line,) 
 
    # Создание анимации 
    anim = FuncAnimation(fig, animate, frames=len(x_line)+1, interval=50) 
 
    # Сохранение анимации в файл 
    writer = PillowWriter(fps=20) 
    anim.save('request_3.gif', writer=writer) 
 
    # Показываем график 
    plt.show()

 
def additional_request(): 
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,6))


    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2, 2)

    plt.title("Funtion y = sin(x)", color="cyan")
    plt.xlabel("X", color="cyan")
    plt.ylabel("Y", color="cyan")

    line, = plt.plot([], [])

    def animate(frame):
        x = np.linspace(-2*np.pi, 2*np.pi)
        y = np.sin(x  - frame/15)
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, frames=200, interval=10, blit=True)
    writer = PillowWriter(fps=20)
    anim.save('additional_request.gif', writer=writer)
    plt.show() 
 
 
request_1() 
request_2() 
request_3() 
additional_request()