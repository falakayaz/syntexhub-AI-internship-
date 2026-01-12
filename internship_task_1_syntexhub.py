import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Manhattan heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, goal):
    rows, cols = maze.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    visited = []

    while open_set:
        _, current = heapq.heappop(open_set)
        visited.append(current)

        if current == goal:
            return reconstruct_path(came_from, current), visited

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze[neighbor] == 1:
                    continue

                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

    return None, visited

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Maze: 0 = free, 1 = wall
maze = np.array([
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

start = (0, 0)
goal = (4, 4)

path, visited = astar(maze, start, goal)

fig, ax = plt.subplots()
ax.set_title("A* Maze Solver Animation")

def update(frame):
    ax.clear()
    ax.imshow(maze, cmap="gray_r")

    for v in visited[:frame]:
        ax.plot(v[1], v[0], "bo", markersize=6)

    if path:
        for p in path:
            ax.plot(p[1], p[0], "r*", markersize=8)

    ax.plot(start[1], start[0], "gs", markersize=10)
    ax.plot(goal[1], goal[0], "rs", markersize=10)

    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(fig, update, frames=len(visited), interval=300)
plt.show()
