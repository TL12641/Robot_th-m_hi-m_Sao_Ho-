"""CSP: Backtracking và Forward Checking để tối ưu thứ tự thu mẫu vật.

Bài toán CSP: gán thứ tự thu mẫu sao cho tổng đường đi là ngắn nhất.
Biến: vị trí thứ i trong hành trình (i = 0..5)
Domain: tập các mẫu chưa được gán
Ràng buộc: tổng đường đi không vượt ngưỡng cho phép
"""

from collections import deque
from utils import get_neighbors, manhattan, get_cost


def _bfs_cost(grid, start, goal):
    """BFS tính chi phí đường đi thực tế."""
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
    return float('inf')


def _bfs_path(grid, start, goal):
    """BFS trả về đường đi đầy đủ."""
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


def _build_result(grid, start, order, base):
    """Xây dựng kết quả đầy đủ từ thứ tự đã tối ưu."""
    full_path = [start]
    total_steps = 0
    total_cost  = 0
    current = start

    for target in list(order) + [base]:
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


# --- Backtracking ---

def _bt_search(grid, start, remaining, base, current_pos, current_cost, threshold, best):
    """Đệ quy backtracking: thử từng mẫu, cắt nhánh nếu vượt ngưỡng."""
    if not remaining:
        # Tính chi phí về base
        total = current_cost + _bfs_cost(grid, current_pos, base)
        if total < best["cost"]:
            best["cost"] = total
            best["order"] = list(best["path"])
        return

    for sample in remaining:
        cost_to = _bfs_cost(grid, current_pos, sample)
        if current_cost + cost_to >= best["cost"]:
            continue  # Cắt nhánh

        # Ước lượng chi phí còn lại (lower bound)
        new_remaining = [s for s in remaining if s != sample]
        lb = sum(
            min(manhattan(sample, s) for s in new_remaining + [base])
            for s in new_remaining
        ) if new_remaining else 0

        if current_cost + cost_to + lb >= best["cost"]:
            continue  # Cắt nhánh bằng lower bound

        best["path"].append(sample)
        _bt_search(grid, start, new_remaining, base,
                   sample, current_cost + cost_to, threshold, best)
        best["path"].pop()


def backtracking(grid, start, samples, base):
    """Backtracking: thử mọi thứ tự, cắt nhánh khi vượt chi phí tốt nhất."""
    # Khởi tạo nghiệm ban đầu bằng greedy nearest
    greedy_order = []
    remaining = list(samples)
    curr = start
    while remaining:
        nearest = min(remaining, key=lambda s: _bfs_cost(grid, curr, s))
        greedy_order.append(nearest)
        remaining.remove(nearest)
        curr = nearest

    init_cost = sum(
        _bfs_cost(grid, ([start] + greedy_order)[i], greedy_order[i])
        for i in range(len(greedy_order))
    ) + _bfs_cost(grid, greedy_order[-1], base)

    best = {"cost": init_cost, "order": greedy_order[:], "path": []}
    _bt_search(grid, start, list(samples), base, start, 0, init_cost, best)

    return _build_result(grid, start, best["order"], base)


# --- Forward Checking ---

def _fc_search(grid, start, assignment, domains, base, current_pos, current_cost, best):
    """Đệ quy forward checking: sau mỗi gán, cập nhật domain của biến còn lại."""
    unassigned = [s for s in domains if s not in assignment]

    if not unassigned:
        total = current_cost + _bfs_cost(grid, current_pos, base)
        if total < best["cost"]:
            best["cost"] = total
            best["order"] = list(assignment)
        return

    # Chọn biến theo MRV: domain nhỏ nhất (ở đây domain luôn = unassigned nên chọn theo manhattan)
    var = min(unassigned, key=lambda s: manhattan(current_pos, s))

    for value in list(domains[var]):
        cost_to = _bfs_cost(grid, current_pos, value)
        if current_cost + cost_to >= best["cost"]:
            continue

        # Forward checking: loại bỏ giá trị trong domain của biến khác
        # (ở đây đơn giản hóa: đánh dấu đã dùng)
        new_domains = {k: set(v) for k, v in domains.items()}
        for other in unassigned:
            if other != value:
                new_domains[other].discard(value)

        assignment.append(value)
        _fc_search(grid, start, assignment, new_domains, base,
                   value, current_cost + cost_to, best)
        assignment.pop()


def forward_checking(grid, start, samples, base):
    """Forward Checking: CSP với kiểm tra domain sau mỗi bước gán."""
    # Khởi tạo domain cho mỗi biến = tập tất cả mẫu
    domains = {s: set(samples) for s in samples}

    # Nghiệm khởi tạo bằng greedy
    greedy_order = []
    remaining = list(samples)
    curr = start
    while remaining:
        nearest = min(remaining, key=lambda s: _bfs_cost(grid, curr, s))
        greedy_order.append(nearest)
        remaining.remove(nearest)
        curr = nearest

    init_cost = sum(
        _bfs_cost(grid, ([start] + greedy_order)[i], greedy_order[i])
        for i in range(len(greedy_order))
    ) + _bfs_cost(grid, greedy_order[-1], base)

    best = {"cost": init_cost, "order": greedy_order[:]}
    _fc_search(grid, start, [], domains, base, start, 0, best)

    return _build_result(grid, start, best["order"], base)
