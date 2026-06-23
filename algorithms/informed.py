"""Thuật toán tìm kiếm có thông tin: Greedy Best-First và A*."""

import heapq
from utils import get_neighbors, manhattan, get_cost


def _reconstruct_path(came_from, start, goal):
    """Truy ngược đường đi từ goal về start."""
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    path.reverse()
    return path


def _find_path_greedy(grid, start, goal):
    """Greedy: ưu tiên ô có heuristic nhỏ nhất (gần goal nhất). Trả về (path, visited)."""
    # heap: (heuristic, node)
    heap = [(manhattan(start, goal), start)]
    came_from = {start: None}
    visited_order = []

    while heap:
        _, curr = heapq.heappop(heap)
        if curr in visited_order:
            continue
        visited_order.append(curr)

        if curr == goal:
            return _reconstruct_path(came_from, start, goal), visited_order

        for nb in get_neighbors(grid, curr):
            if nb not in came_from:
                came_from[nb] = curr
                heapq.heappush(heap, (manhattan(nb, goal), nb))

    return [], visited_order


def _find_path_astar(grid, start, goal):
    """A*: f(n) = g(n) + h(n), đảm bảo tìm đường ngắn nhất. Trả về (path, visited)."""
    # heap: (f, g, node)
    heap = [(manhattan(start, goal), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    visited_order = []

    while heap:
        f, g, curr = heapq.heappop(heap)

        if curr in visited_order:
            continue
        visited_order.append(curr)

        if curr == goal:
            return _reconstruct_path(came_from, start, goal), visited_order

        for nb in get_neighbors(grid, curr):
            new_g = g + get_cost(grid, nb)  # Chi phí thực tế (ROUGH = 3)
            if nb not in g_score or new_g < g_score[nb]:
                g_score[nb] = new_g
                came_from[nb] = curr
                f_new = new_g + manhattan(nb, goal)
                heapq.heappush(heap, (f_new, new_g, nb))

    return [], visited_order


def _run_algo(grid, start, samples, base, find_path_fn):
    """Chạy thuật toán theo thứ tự mẫu gần nhất, rồi về base."""
    remaining = list(samples)
    current = start
    full_path = [start]
    full_visited = []
    total_steps = 0
    total_cost  = 0  # Chi phí thực tính ROUGH×5

    while remaining:
        nearest = min(remaining, key=lambda s: manhattan(current, s))
        remaining.remove(nearest)

        path, visited = find_path_fn(grid, current, nearest)
        if not path:
            return {"path": [], "visited": full_visited,
                    "stats": {"steps": 0, "cost": 0, "visited": len(full_visited), "found": False}}

        full_path.extend(path[1:])
        full_visited.extend(visited)
        total_steps += len(path) - 1
        total_cost  += sum(get_cost(grid, p) for p in path[1:])
        current = nearest

    path, visited = find_path_fn(grid, current, base)
    if path:
        full_path.extend(path[1:])
        full_visited.extend(visited)
        total_steps += len(path) - 1
        total_cost  += sum(get_cost(grid, p) for p in path[1:])

    return {
        "path": full_path,
        "visited": full_visited,
        "stats": {"steps": total_steps, "cost": total_cost, "visited": len(full_visited), "found": True}
    }


def greedy(grid, start, samples, base):
    """Greedy Best-First: chọn ô gần goal nhất theo heuristic Manhattan."""
    return _run_algo(grid, start, samples, base, _find_path_greedy)


def astar(grid, start, samples, base):
    """A*: kết hợp chi phí thực g(n) và heuristic h(n) để tìm đường tối ưu."""
    return _run_algo(grid, start, samples, base, _find_path_astar)
