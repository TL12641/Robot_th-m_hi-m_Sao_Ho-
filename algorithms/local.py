"""Thuật toán tìm kiếm cục bộ: Hill Climbing và Simulated Annealing.

Các thuật toán này tối ưu THỨ TỰ thu mẫu vật, sau đó dùng BFS để tìm đường.
"""

import math
import random
from collections import deque
from utils import get_neighbors, manhattan, get_cost


def _bfs_path_cost(grid, start, goal):
    """BFS tìm chi phí đường đi thực tế từ start đến goal."""
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        curr, cost = queue.popleft()
        if curr == goal:
            return cost
        for nb in get_neighbors(grid, curr):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, cost + 1))
    return float('inf')  # Không tìm thấy


def _bfs_path(grid, start, goal):
    """BFS trả về đường đi đầy đủ từ start đến goal."""
    queue = deque([start])
    came_from = {start: None}
    while queue:
        curr = queue.popleft()
        if curr == goal:
            path = []
            while curr is not None:
                path.append(curr)
                curr = came_from[curr]
            path.reverse()
            return path
        for nb in get_neighbors(grid, curr):
            if nb not in came_from:
                came_from[nb] = curr
                queue.append(nb)
    return []


def _total_cost(grid, start, order, base):
    """Tính tổng chi phí đường đi theo một thứ tự thu mẫu cụ thể."""
    points = [start] + list(order) + [base]
    total = 0
    for i in range(len(points) - 1):
        total += _bfs_path_cost(grid, points[i], points[i + 1])
    return total


def _build_result(grid, start, order, base):
    """Xây dựng kết quả đầy đủ theo thứ tự đã chọn."""
    full_path = [start]
    total_steps = 0
    total_cost  = 0
    current = start
    points = list(order) + [base]

    for target in points:
        path = _bfs_path(grid, current, target)
        if not path:
            return {"path": [], "visited": [], "stats": {"steps": 0, "cost": 0, "visited": 0, "found": False}}
        full_path.extend(path[1:])
        total_steps += len(path) - 1
        total_cost  += sum(get_cost(grid, p) for p in path[1:])
        current = target

    return {
        "path": full_path,
        "visited": full_path,
        "stats": {"steps": total_steps, "cost": total_cost, "visited": total_steps, "found": True}
    }


def hill_climbing(grid, start, samples, base):
    """Hill Climbing: hoán đổi 2 mẫu ngẫu nhiên, giữ nếu tổng đường đi ngắn hơn."""
    order = list(samples)
    random.shuffle(order)
    current_cost = _total_cost(grid, start, order, base)

    # Thử tất cả cặp hoán đổi, lặp cho đến khi không cải thiện được
    improved = True
    while improved:
        improved = False
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                new_order = order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_cost = _total_cost(grid, start, new_order, base)
                if new_cost < current_cost:
                    order = new_order
                    current_cost = new_cost
                    improved = True

    return _build_result(grid, start, order, base)


def simulated_annealing(grid, start, samples, base):
    """SA: chấp nhận nghiệm xấu hơn theo xác suất e^(-ΔE/T), T giảm dần."""
    order = list(samples)
    random.shuffle(order)
    current_cost = _total_cost(grid, start, order, base)
    best_order = order[:]
    best_cost = current_cost

    # Tham số SA
    T = 1000.0       # Nhiệt độ ban đầu
    T_min = 0.1      # Nhiệt độ dừng
    alpha = 0.995    # Tốc độ làm nguội

    while T > T_min:
        # Hoán đổi 2 mẫu ngẫu nhiên
        i, j = random.sample(range(len(order)), 2)
        new_order = order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_cost = _total_cost(grid, start, new_order, base)

        delta = new_cost - current_cost
        # Chấp nhận nếu tốt hơn, hoặc theo xác suất nếu xấu hơn
        if delta < 0 or random.random() < math.exp(-delta / T):
            order = new_order
            current_cost = new_cost

        if current_cost < best_cost:
            best_order = order[:]
            best_cost = current_cost

        T *= alpha  # Làm nguội

    return _build_result(grid, start, best_order, base)
