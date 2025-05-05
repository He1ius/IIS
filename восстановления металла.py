import numpy as np
import math
import random

def target_function(x, y):
    """Целевая функция для максимизации"""
    return 1 / (1 + x**2 + y**2)

def simulated_annealing(target_func, initial_temp, final_temp, cooling_rate, max_iter):
    """
    Алгоритм имитации отжига
    
    Параметры:
    - target_func: целевая функция
    - initial_temp: начальная температура
    - final_temp: конечная температура
    - cooling_rate: скорость охлаждения
    - max_iter: максимальное число итераций
    
    Возвращает:
    - Лучшее найденное решение (x, y)
    - Значение функции в этой точке
    - Историю значений для визуализации
    """
    # Начальное случайное решение
    current_x = random.uniform(-10, 10)
    current_y = random.uniform(-10, 10)
    current_value = target_func(current_x, current_y)
    
    best_x, best_y = current_x, current_y
    best_value = current_value
    
    temp = initial_temp
    history = []
    
    iteration = 0
    while temp > final_temp and iteration < max_iter:
        # Генерируем соседнее решение
        neighbor_x = current_x + random.uniform(-1, 1)
        neighbor_y = current_y + random.uniform(-1, 1)
        neighbor_value = target_func(neighbor_x, neighbor_y)
        
        # Разница значений
        delta = neighbor_value - current_value
        
        # Если решение лучше, всегда принимаем его
        if delta > 0:
            current_x, current_y = neighbor_x, neighbor_y
            current_value = neighbor_value
            
            # Обновляем лучшее решение
            if current_value > best_value:
                best_x, best_y = current_x, current_y
                best_value = current_value
        else:
            # Если решение хуже, принимаем его с некоторой вероятностью
            probability = math.exp(delta / temp)
            if random.random() < probability:
                current_x, current_y = neighbor_x, neighbor_y
                current_value = neighbor_value
        
        # Сохраняем историю для визуализации
        history.append((current_x, current_y, current_value, temp))
        
        # Охлаждаем систему
        temp *= cooling_rate
        iteration += 1
    
    return (best_x, best_y), best_value, history

# Параметры алгоритма
initial_temp = 1000
final_temp = 0.1
cooling_rate = 0.95
max_iter = 10000

# Запуск алгоритма
(best_x, best_y), best_value, history = simulated_annealing(
    target_function, initial_temp, final_temp, cooling_rate, max_iter
)

# Вывод результатов
print("Результаты оптимизации методом имитации отжига:")
print(f"Лучшее решение: x = {best_x:.4f}, y = {best_y:.4f}")
print(f"Максимальное значение функции: {best_value:.6f}")
print(f"Количество итераций: {len(history)}")

# Визуализация процесса (требует matplotlib)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Подготовка данных для 3D графика функции
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = target_function(X, Y)
    
    # Траектория поиска
    path_x = [h[0] for h in history]
    path_y = [h[1] for h in history]
    path_z = [h[2] for h in history]
    
    # Создание 3D графика
    fig = plt.figure(figsize=(12, 8))
    
    # График функции
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.scatter(path_x, path_y, path_z, c='r', s=5, label='Поиск')
    ax1.scatter([best_x], [best_y], [best_value], c='black', s=100, marker='*', label='Оптимум')
    ax1.set_title('Функция и процесс поиска')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x, y)')
    ax1.legend()
    
    # График сходимости
    ax2 = fig.add_subplot(122)
    ax2.plot([h[2] for h in history], 'r-')
    ax2.set_title('Сходимость алгоритма')
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('Значение функции')
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\nДля визуализации требуется установить matplotlib: pip install matplotlib")
