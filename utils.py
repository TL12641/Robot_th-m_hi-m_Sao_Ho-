"""Các tiện ích dùng chung: sinh bản đồ, heuristic, kiểm tra đường đi."""

import random
from collections import deque

# Hằng số ô bản đồ
EMPTY = 0
WALL = 1
SAMPLE = 2
BASE = 3
ROUGH = 4  # Địa hình gồ ghề, chi phí đi vào = 3

# Chi phí đi vào từng loại ô
CELL_COST = {
    EMPTY: 1,
    ROUGH: 5,
    SAMPLE: 1,
    BASE: 1,
}

# Kích thước bản đồ
GRID_SIZE = 15
NUM_SAMPLES = 6
WALL_RATIO = 0.15   # 15% tường
ROUGH_RATIO = 0.12  # 12% địa hình gồ ghề

# 4 hướng di chuyển: lên, xuống, trái, phải
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_cost(grid, pos):
    """Trả về chi phí đi vào ô pos (ROUGH = 3, còn lại = 1)."""
    return CELL_COST.get(grid[pos[0]][pos[1]], 1)


def manhattan(a, b):
    """Tính khoảng cách Manhattan giữa 2 điểm."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_valid(grid, row, col):
    """Kiểm tra ô (row, col) có hợp lệ và không phải tường không."""
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and grid[row][col] != WALL


def get_neighbors(grid, pos):
    """Trả về danh sách ô kề hợp lệ của pos."""
    neighbors = []
    for dr, dc in DIRECTIONS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if is_valid(grid, nr, nc):
            neighbors.append((nr, nc))
    return neighbors


def is_reachable(grid, start, target):
    """Kiểm tra target có đến được từ start bằng BFS không."""
    visited = {start}
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        if curr == target:
            return True
        for nb in get_neighbors(grid, curr):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return False


def path_cost(grid, points):
    """Tính tổng chi phí đường đi qua danh sách điểm (dùng Manhattan)."""
    total = 0
    for i in range(len(points) - 1):
        total += manhattan(points[i], points[i + 1])
    return total


def generate_map(seed=None):
    """
    Sinh bản đồ 15×15 ngẫu nhiên với chướng ngại vật, 6 mẫu vật và 1 base.
    Đảm bảo tất cả mẫu vật và base đều có thể đến được từ vị trí robot ban đầu.
    Trả về: (grid, robot_start, samples, base)
    """
    rng = random.Random(seed)

    while True:
        # Tạo lưới trống
        grid = [[EMPTY] * GRID_SIZE for _ in range(GRID_SIZE)]

        # Đặt tường ngẫu nhiên
        num_walls = int(GRID_SIZE * GRID_SIZE * WALL_RATIO)
        wall_cells = rng.sample(
            [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)],
            num_walls
        )
        for r, c in wall_cells:
            grid[r][c] = WALL

        # Lấy tất cả ô trống
        empty_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
                       if grid[r][c] == EMPTY]

        if len(empty_cells) < NUM_SAMPLES + 2:
            continue  # Không đủ ô trống, thử lại

        # Chọn vị trí robot, base và các mẫu vật
        chosen = rng.sample(empty_cells, NUM_SAMPLES + 2)
        robot_start = chosen[0]
        base = chosen[1]
        samples = chosen[2:]

        # Đặt địa hình gồ ghề (ROUGH) ngẫu nhiên lên ô trống còn lại
        remaining_empty = [c for c in empty_cells if c not in chosen]
        num_rough = int(GRID_SIZE * GRID_SIZE * ROUGH_RATIO)
        rough_cells = rng.sample(remaining_empty, min(num_rough, len(remaining_empty)))
        for r, c in rough_cells:
            grid[r][c] = ROUGH

        # Đặt mẫu vật và base lên lưới
        for r, c in samples:
            grid[r][c] = SAMPLE
        grid[base[0]][base[1]] = BASE

        # Kiểm tra tất cả đều reachable từ robot_start
        all_targets = samples + [base]
        if all(is_reachable(grid, robot_start, t) for t in all_targets):
            return grid, robot_start, list(samples), base
