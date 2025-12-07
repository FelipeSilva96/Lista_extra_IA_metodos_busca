import heapq
import time
import math
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, RadioButtons


# ==========================================
# Estrutura de Nó e Heurísticas (Inalterado)
# ==========================================
class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f


class Heuristics:
    @staticmethod
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def euclidean(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def chebyshev(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# ==========================================
# Motor de Busca (Inalterado)
# ==========================================
class MazeSolver:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.start = self._find_pos(2)
        self.end = self._find_pos(3)

    def _find_pos(self, value):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze[r][c] == value:
                    return (r, c)
        return (0, 0)

    def get_neighbors(self, node):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = node.position[0] + dr, node.position[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.maze[nr][nc] != 1:
                neighbors.append((nr, nc))
        return neighbors

    def reconstruct_path(self, node):
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]

    def solve_bfs(self):
        start_time = time.time()
        if not self.start or not self.end:
            return None

        # Inicializa o nó de partida com custo 0
        queue = deque([Node(self.start, None, 0, 0)])
        visited = {self.start}
        nodes_visited = 0

        while queue:
            current = queue.popleft()
            nodes_visited += 1

            if current.position == self.end:
                return self._build_result(current, nodes_visited, start_time, "BFS")

            for neighbor_pos in self.get_neighbors(current):
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    # CORREÇÃO AQUI: Passamos current.g + 1 para o novo nó
                    new_node = Node(neighbor_pos, current, current.g + 1, 0)
                    queue.append(new_node)

        return None

    def solve_uniform_cost(self):
        start_time = time.time()
        if not self.start or not self.end:
            return None
        open_list = [Node(self.start, None, 0, 0)]
        visited = set()
        nodes_visited = 0

        while open_list:
            current = heapq.heappop(open_list)
            nodes_visited += 1
            if current.position == self.end:
                return self._build_result(
                    current, nodes_visited, start_time, "Custo Uniforme"
                )
            if current.position in visited:
                continue
            visited.add(current.position)
            for neighbor_pos in self.get_neighbors(current):
                if neighbor_pos not in visited:
                    new_node = Node(neighbor_pos, current, current.g + 1, 0)
                    heapq.heappush(open_list, new_node)
        return None

    def solve_astar(self, heuristic_func, heuristic_name):
        start_time = time.time()
        if not self.start or not self.end:
            return None
        start_node = Node(self.start, None, 0, heuristic_func(self.start, self.end))
        open_list = [start_node]
        visited = set()
        nodes_visited = 0

        while open_list:
            current = heapq.heappop(open_list)
            nodes_visited += 1
            if current.position == self.end:
                return self._build_result(
                    current, nodes_visited, start_time, f"A* ({heuristic_name})"
                )
            if current.position in visited:
                continue
            visited.add(current.position)
            for neighbor_pos in self.get_neighbors(current):
                if neighbor_pos not in visited:
                    g = current.g + 1
                    h = heuristic_func(neighbor_pos, self.end)
                    new_node = Node(neighbor_pos, current, g, h)
                    heapq.heappush(open_list, new_node)
        return None

    def _build_result(self, node, nodes_visited, start_time, name):
        return {
            "algorithm": name,
            "path": self.reconstruct_path(node),
            "cost": node.g,
            "nodes_visited": nodes_visited,
            "time": (time.time() - start_time) * 1000,
        }


# ==========================================
# Geração e Execução
# ==========================================
def generate_solvable_maze(rows=15, cols=15):
    # Ajusta para ímpar para garantir paredes bonitas ao redor dos corredores
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    # 1. Começa preenchido de paredes (1)
    maze = [[1 for _ in range(cols)] for _ in range(rows)]

    # Função auxiliar para pegar vizinhos a 2 células de distância
    def get_neighbors(r, c):
        candidates = [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]
        valid = []
        for nr, nc in candidates:
            if 0 <= nr < rows and 0 <= nc < cols:
                valid.append((nr, nc))
        random.shuffle(valid)
        return valid

    # 2. Algoritmo de Escavação (Recursive Backtracking)
    start_r, start_c = 0, 0
    maze[start_r][start_c] = 0
    stack = [(start_r, start_c)]

    while stack:
        current_r, current_c = stack[-1]
        neighbors = get_neighbors(current_r, current_c)
        found_path = False

        for nr, nc in neighbors:
            if maze[nr][nc] == 1:  # Se ainda é parede virgem
                # Derruba a parede entre o atual e o vizinho
                wall_r = (current_r + nr) // 2
                wall_c = (current_c + nc) // 2
                maze[wall_r][wall_c] = 0
                maze[nr][nc] = 0

                stack.append((nr, nc))
                found_path = True
                break

        if not found_path:
            stack.pop()

    # 3. Criar Loops (Braiding)
    # Remove paredes extras aleatoriamente para criar múltiplos caminhos.
    # Sem isso, o labirinto seria "perfeito" (apenas 1 caminho possível),
    # o que torna a comparação de heurísticas inútil.
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if maze[r][c] == 1:
                # 15% de chance de remover uma parede interna
                if random.random() < 0.15:
                    # Evita abrir paredes que conectam nada a nada (opcional)
                    neighbors_count = 0
                    if maze[r + 1][c] == 0:
                        neighbors_count += 1
                    if maze[r - 1][c] == 0:
                        neighbors_count += 1
                    if maze[r][c + 1] == 0:
                        neighbors_count += 1
                    if maze[r][c - 1] == 0:
                        neighbors_count += 1

                    if neighbors_count >= 2:
                        maze[r][c] = 0

    # Define Início e Fim
    maze[0][0] = 2
    maze[rows - 1][cols - 1] = 3

    return maze


def run_suite(maze):
    solver = MazeSolver(maze)
    # Dicionário para acesso rápido pelo nome
    results = {}

    res = solver.solve_bfs()
    if res:
        results["BFS"] = res

    res = solver.solve_uniform_cost()
    if res:
        results["Uniforme"] = res

    res = solver.solve_astar(Heuristics.manhattan, "Manhattan")
    if res:
        results["A* (Manhattan)"] = res

    res = solver.solve_astar(Heuristics.euclidean, "Euclidiana")
    if res:
        results["A* (Euclidiana)"] = res

    res = solver.solve_astar(Heuristics.chebyshev, "Chebyshev")
    if res:
        results["A* (Chebyshev)"] = res

    return results


# ==========================================
# [RF010] Interface Gráfica Modificada
# ==========================================
class MazeViewer:
    def __init__(self):
        self.rows = 30
        self.cols = 30

        # Estado inicial
        self.maze = generate_solvable_maze(self.rows, self.cols)
        self.results = run_suite(self.maze)
        self.current_algo_name = "BFS"  # Seleção padrão
        self.show_path_flag = False  # Oculto por padrão

        # Configuração da Figura
        # Aumentamos a largura para caber o menu lateral
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(left=0.3, bottom=0.2)

        self.cmap = mcolors.ListedColormap(["white", "black", "green", "red", "blue"])
        self.norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], self.cmap.N)

        # --- WIDGETS ---

        # 1. MENU DE SELEÇÃO (Radio Buttons)
        ax_radio = plt.axes([0.02, 0.4, 0.20, 0.3])  # Posição lateral esquerda
        ax_radio.set_facecolor("#f0f0f0")
        self.radio = RadioButtons(ax_radio, list(self.results.keys()))
        self.radio.on_clicked(self.change_algorithm)

        # 2. BOTÃO "MOSTRAR CAMINHO"
        ax_show = plt.axes([0.35, 0.05, 0.25, 0.075])
        self.btn_show = Button(
            ax_show, "Mostrar Caminho", color="lightblue", hovercolor="skyblue"
        )
        self.btn_show.on_clicked(self.reveal_path)

        # 3. BOTÃO "NOVO LABIRINTO"
        ax_new = plt.axes([0.65, 0.05, 0.25, 0.075])
        self.btn_new = Button(
            ax_new, "Novo Labirinto", color="lightgreen", hovercolor="lime"
        )
        self.btn_new.on_clicked(self.regenerate_maze)

        self.plot_current()
        plt.show()

    def plot_current(self):
        self.ax.clear()

        # Recupera resultado atual
        result = self.results.get(self.current_algo_name)
        if not result:
            return

        # Prepara a matriz visual
        maze_copy = [row[:] for row in self.maze]

        # SÓ DESENHA O CAMINHO SE A FLAG ESTIVER ATIVADA
        if self.show_path_flag:
            for r, c in result["path"]:
                if maze_copy[r][c] not in (2, 3):
                    maze_copy[r][c] = 4

            # Mostra métricas reais
            title_text = (
                f"Algoritmo: {result['algorithm']}\n"
                f"Custo: {result['cost']} | Nós Visitados: {result['nodes_visited']}\n"
                f"Tempo: {result['time']:.4f}ms"
            )
        else:
            # Oculta métricas para não dar spoiler do desempenho
            title_text = (
                f"Algoritmo: {result['algorithm']}\n"
                f"Caminho Oculto (Clique em 'Mostrar Caminho')"
            )

        self.ax.imshow(maze_copy, cmap=self.cmap, norm=self.norm)
        self.ax.grid(which="major", color="gray", linestyle="-", linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title_text)

        self.fig.canvas.draw()

    def change_algorithm(self, label):
        """Chamado quando o usuário seleciona um item no menu"""
        self.current_algo_name = label
        self.show_path_flag = False  # Reseta a visualização ao trocar de algoritmo
        self.plot_current()

    def reveal_path(self, event):
        """Chamado pelo botão Mostrar Caminho"""
        self.show_path_flag = True
        self.plot_current()

    def regenerate_maze(self, event):
        """Gera novo labirinto e reseta tudo"""
        print("Gerando novo labirinto...")
        self.maze = generate_solvable_maze(self.rows, self.cols)
        self.results = run_suite(self.maze)

        # Atualiza a lista de algoritmos no menu (caso algum tenha falhado, embora improvável)
        # Nota: Matplotlib RadioButtons não atualiza labels facilmente,
        # mas como os algoritmos são fixos, não precisamos recriar o widget.

        self.show_path_flag = False
        # Mantém o algoritmo selecionado atualmente se possível
        if self.current_algo_name not in self.results:
            self.current_algo_name = list(self.results.keys())[0]

        self.plot_current()


if __name__ == "__main__":
    app = MazeViewer()
