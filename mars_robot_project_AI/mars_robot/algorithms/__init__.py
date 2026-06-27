# __init__.py
# File nay export tat ca cac thuat toan ra ben ngoai
# De main.py co the import va su dung
#
# Co 5 nhom thuat toan:
#   Nhom 1: Tim kiem mu (BFS, DFS)
#   Nhom 2: Tim kiem co thong tin (Greedy, A*)
#   Nhom 3: Tim kiem cuc bo (Hill Climbing, Simulated Annealing)
#   Nhom 4: CSP (Backtracking, Forward Checking)
#   Nhom 5: Khong xac dinh (AND-OR, Sensorless)
#   Nhom 6: Doi khang (Minimax, Alpha-Beta)

from .uninformed    import bfs, dfs
from .informed      import greedy, astar
from .local         import hill_climbing, simulated_annealing
from .csp           import backtracking, forward_checking
from .nondeterministic import and_or_search, sensorless_search
from .adversarial   import minimax, alpha_beta

# Nhom 1+2: Tim kiem duong di (Pathfinding)
# BFS = Breadth-First, DFS = Depth-First, Greedy, A*
PATHFINDING_ALGOS = {
    "BFS":    bfs,
    "DFS":    dfs,
    "Greedy": greedy,
    "A*":     astar,
}

# Nhom 3+4: Toi uu thu tu thu mau (Order Optimization)
# Khac nhom 1,2: tim thu tu thu mau toi uu, khong phai tim duong thang
ORDER_ALGOS = {
    "Hill Climbing":    hill_climbing,
    "Sim. Annealing":   simulated_annealing,
    "Backtracking":     backtracking,
    "Forward Checking": forward_checking,
}

# Nhom 5: Khong xac dinh (Non-deterministic)
# AND-OR: ke hoach co dieu kien khi co truot
# Sensorless: khong biet vi tri ban dau → thu hep belief state
NONDETERMINISTIC_ALGOS = {
    "AND-OR":    and_or_search,
    "Sensorless": sensorless_search,
}

# Nhom 6: Doi khang (Adversarial)
# 2 robot thi thu mau, Robot A muon thang, Robot B muon can tra
ADVERSARIAL_ALGOS = {
    "Minimax":    minimax,
    "Alpha-Beta": alpha_beta,
}

# Gop tat ca vao 1 dict → de dung trong main.py
# Dung vong for thay vi ** unpacking cho de hieu hon
ALL_ALGOS = {}

for ten, ham in PATHFINDING_ALGOS.items():
    ALL_ALGOS[ten] = ham

for ten, ham in ORDER_ALGOS.items():
    ALL_ALGOS[ten] = ham

for ten, ham in NONDETERMINISTIC_ALGOS.items():
    ALL_ALGOS[ten] = ham

for ten, ham in ADVERSARIAL_ALGOS.items():
    ALL_ALGOS[ten] = ham
