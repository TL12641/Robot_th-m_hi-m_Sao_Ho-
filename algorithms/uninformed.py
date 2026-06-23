"""Thuật toán tìm kiếm không có thông tin: BFS và DFS."""

from collections import deque
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


def _find_path_bfs(grid, start, goal):
    """BFS tìm đường ngắn nhất từ start đến goal. Trả về (path, visited)."""
    queue = deque([start])
    came_from = {start: None}
    visited_order = []

    while queue:
        curr = queue.popleft()
        visited_order.append(curr)

        if curr == goal:
            path = _reconstruct_path(came_from, start, goal)
            return path, visited_order

        for nb in get_neighbors(grid, curr):
            if nb not in came_from:
                came_from[nb] = curr
                queue.append(nb)

    return [], visited_order  # Không tìm thấy đường


def _find_path_dfs(grid, start, goal):
    """DFS tìm đường từ start đến goal (không tối ưu). Trả về (path, visited)."""
    stack = [start]
    came_from = {start: None}
    visited_order = []

    while stack:
        curr = stack.pop()
        if curr in visited_order:
            continue
        visited_order.append(curr)

        if curr == goal:
            path = _reconstruct_path(came_from, start, goal)
            return path, visited_order

        for nb in get_neighbors(grid, curr):
            if nb not in came_from:
                came_from[nb] = curr
                stack.append(nb)

    return [], visited_order


def _run_algo(grid, start, samples, base, find_path_fn):
    """
    Chạy thuật toán tìm đường theo thứ tự mẫu vật gần nhất.
    Trả về dict chuẩn: path, visited, stats.
    """
    remaining = list(samples)
    current = start
    full_path = [start]
    full_visited = []
    total_steps = 0
    total_cost  = 0  # Chi phí thực tính ROUGH×5

    # Thu mẫu theo thứ tự gần nhất (greedy nearest)
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

    # Về base
    path, visited = find_path_fn(grid, current, base)
    if path:
        full_path.extend(path[1:])
        full_visited.extend(visited)
        total_steps += len(path) - 1
        total_cost  += sum(get_cost(grid, p) for p in path[1:])

    return {
        "path": full_path,
        "visited": full_visited,
        "stats": {
            "steps": total_steps,
            "cost":  total_cost,
            "visited": len(full_visited),
            "found": True,
        }
    }


def bfs(grid, start, samples, base):
    """BFS: tìm đường ngắn nhất, duyệt theo từng lớp."""
    return _run_algo(grid, start, samples, base, _find_path_bfs)


def dfs(grid, start, samples, base):
    """DFS: tìm đường bằng stack, không đảm bảo tối ưu."""
    return _run_algo(grid, start, samples, base, _find_path_dfs)
