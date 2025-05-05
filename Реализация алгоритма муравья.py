import numpy as np
import random
from math import sqrt, exp

class AntColonyTSP:
    def __init__(self, distances, n_ants=10, n_iterations=100, decay=0.5, alpha=1, beta=2):
        """
        Инициализация параметров алгоритма:
        - distances: матрица расстояний между городами
        - n_ants: количество муравьев
        - n_iterations: количество итераций
        - decay: коэффициент испарения феромонов
        - alpha: вес феромонов при выборе пути
        - beta: вес расстояния при выборе пути
        """
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.n_cities = len(distances)
        self.pheromones = np.ones((self.n_cities, self.n_cities)) / self.n_cities

    def run(self):
        best_path = None
        best_distance = float('inf')
        
        for _ in range(self.n_iterations):
            paths = self._generate_paths()
            self._update_pheromones(paths)
            
            current_best_path, current_best_dist = min(paths, key=lambda x: x[1])
            if current_best_dist < best_distance:
                best_path = current_best_path
                best_distance = current_best_dist
        
        return best_path, best_distance

    def _generate_paths(self):
        paths = []
        for _ in range(self.n_ants):
            path = self._construct_path()
            distance = self._calculate_distance(path)
            paths.append((path, distance))
        return paths

    def _construct_path(self):
        path = []
        visited = set()
        
        # Начинаем со случайного города
        current_city = random.randint(0, self.n_cities - 1)
        path.append(current_city)
        visited.add(current_city)
        
        while len(visited) < self.n_cities:
            next_city = self._select_next_city(current_city, visited)
            path.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        return path

    def _select_next_city(self, current_city, visited):
        probabilities = []
        total = 0
        
        for city in range(self.n_cities):
            if city not in visited:
                pheromone = self.pheromones[current_city][city] ** self.alpha
                heuristic = (1 / self.distances[current_city][city]) ** self.beta
                probabilities.append((city, pheromone * heuristic))
                total += pheromone * heuristic
        
        # Нормализация вероятностей
        probabilities = [(city, prob/total) for city, prob in probabilities]
        
        # Выбор следующего города по вероятностям
        r = random.random()
        cumulative_prob = 0
        for city, prob in probabilities:
            cumulative_prob += prob
            if r <= cumulative_prob:
                return city
        
        return probabilities[-1][0]

    def _calculate_distance(self, path):
        distance = 0
        for i in range(len(path)):
            distance += self.distances[path[i-1]][path[i]] if i > 0 else 0
        return distance

    def _update_pheromones(self, paths):
        # Испарение феромонов
        self.pheromones *= (1 - self.decay)
        
        # Обновление феромонов на основе пройденных путей
        for path, distance in paths:
            for i in range(len(path)):
                city_a = path[i-1] if i > 0 else path[-1]
                city_b = path[i]
                self.pheromones[city_a][city_b] += 1 / distance
                self.pheromones[city_b][city_a] += 1 / distance

# Пример использования
if __name__ == "__main__":
    # Матрица расстояний между городами (пример)
    cities = [(2, 3), (5, 7), (8, 1), (1, 9), (4, 6)]
    n_cities = len(cities)
    
    # Создаем матрицу расстояний
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                dx = cities[i][0] - cities[j][0]
                dy = cities[i][1] - cities[j][1]
                distances[i][j] = sqrt(dx*dx + dy*dy)
    
    # Запуск алгоритма
    aco = AntColonyTSP(distances, n_ants=10, n_iterations=100)
    best_path, best_distance = aco.run()
    
    print("Лучший найденный путь:", best_path)
    print("Длина пути:", best_distance)
