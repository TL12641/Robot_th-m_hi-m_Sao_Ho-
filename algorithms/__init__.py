"""Gói thuật toán tìm đường và tối ưu thứ tự cho Mars Robot."""

from .uninformed import bfs, dfs
from .informed import greedy, astar
from .local import hill_climbing, simulated_annealing
from .csp import backtracking, forward_checking

# Nhóm thuật toán để hiển thị trên UI
PATHFINDING_ALGOS = {
    "BFS": bfs,
    "DFS": dfs,
    "Greedy": greedy,
    "A*": astar,
}

ORDER_ALGOS = {
    "Hill Climbing": hill_climbing,
    "Sim. Annealing": simulated_annealing,
    "Backtracking": backtracking,
    "Forward Checking": forward_checking,
}

ALL_ALGOS = {**PATHFINDING_ALGOS, **ORDER_ALGOS}
