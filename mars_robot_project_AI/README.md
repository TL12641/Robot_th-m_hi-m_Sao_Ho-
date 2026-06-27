# Mars Robot Project AI

## Introduction

Mars Robot Project AI is a Python simulation of an autonomous robot exploring the surface of Mars. The robot must navigate through a randomly generated environment, collect all scientific samples, avoid obstacles, and finally return to the base station.

The project demonstrates the application of Artificial Intelligence search algorithms for pathfinding and task optimization in a simulated exploration environment.

---

## Features

* Randomly generates a Mars map.
* Obstacles and rough terrain.
* Collect multiple scientific samples.
* Return safely to the base.
* Visual simulation using Pygame.
* Compare multiple Artificial Intelligence algorithms.

---

## Implemented Algorithms

### Uninformed Search

* Breadth First Search (BFS)
* Depth First Search (DFS)

### Informed Search

* Greedy Best-First Search
* A* Search

### Local Search

* Hill Climbing
* Simulated Annealing

### Constraint Satisfaction Problem (CSP)

* Backtracking
* Forward Checking

### Other AI Techniques

* Adversarial Search
* Nondeterministic Search

---

## Technologies

* Python 3
* Pygame

---

## Project Structure

```
mars_robot_project_AI/

│── main.py
│── utils.py
│── requirements.txt
│── README.md

└── algorithms/
    ├── __init__.py
    ├── uninformed.py
    ├── informed.py
    ├── local.py
    ├── csp.py
    ├── adversarial.py
    └── nondeterministic.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Authors

Artificial Intelligence Course Project
