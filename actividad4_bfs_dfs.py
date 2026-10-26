"""
ACTIVIDAD 4 (Sección 2.4 Algoritmos de búsqueda ciega: BFS / DFS)
Fundamentos de Inteligencia Artificial

ENUNCIADO
---------
Resolver laberintos con BFS (búsqueda en anchura) y DFS (búsqueda en profundidad)
y comparar: camino solución, longitud, nodos expandidos, tamaño máximo de la
frontera, memoria, tiempo, y su idoneidad según el tipo de problema.

REPRESENTACIÓN DEL PROBLEMA / GRAFO
-----------------------------------
  Laberinto : matriz numpy (1 = pared, 0 = pasillo).
  Grafo     : cada celda libre es un NODO; dos celdas libres adyacentes
              (arriba, derecha, abajo, izquierda) forman una ARISTA no dirigida
              y sin pesos (coste 1 por paso).
  Estructura: LISTA DE ADYACENCIA (diccionario nodo -> lista de vecinos) con orden
              fijo de vecinos: ARRIBA, DERECHA, ABAJO, IZQUIERDA. El orden afecta
              a DFS (qué camino encuentra), no a BFS (siempre el más corto).
  Estado inicial: (1, 1) | Meta: esquina inferior derecha libre.

COMPLEJIDAD (V = nodos, E = aristas, b = ramificación, d = profundidad de la meta
más superficial, m = profundidad máxima)
  BFS : tiempo O(V + E)   espacio O(V)   (en árbol: O(b^d)). Completo; ÓPTIMO en
        número de pasos (grafo sin pesos).
  DFS : tiempo O(V + E)   espacio O(V) con control de visitados (en árbol: O(b·m)).
        Completo en grafos finitos con control de visitados; NO óptimo.

Librerías: numpy, matplotlib, networkx (pip) + collections, dataclasses, random,
           time, tracemalloc, csv, pathlib, typing (biblioteca estándar).
"""

from __future__ import annotations

import csv
import random
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

Nodo = Tuple[int, int]
ORDEN = [(-1, 0), (0, 1), (1, 0), (0, -1)]       # ARRIBA, DERECHA, ABAJO, IZQUIERDA
SEMILLA = 2026


# ----------------------------------------------------------------------------
# 1. GENERACIÓN DE LABERINTOS (datos de entrada)
# ----------------------------------------------------------------------------
def generar_laberinto(celdas_alto: int, celdas_ancho: int, semilla: int,
                      prob_ciclos: float = 0.0) -> np.ndarray:
    """
    Laberinto 'perfecto' (un único camino entre dos celdas) por DFS con retroceso.
    Si prob_ciclos > 0 se derriban paredes interiores al azar creando CICLOS, es
    decir, varios caminos posibles (aquí BFS y DFS se comportan distinto).
    Dimensión de la matriz: (2·alto+1) x (2·ancho+1).
    """
    rng = random.Random(semilla)
    H, W = 2 * celdas_alto + 1, 2 * celdas_ancho + 1
    g = np.ones((H, W), dtype=np.int8)
    g[1, 1] = 0
    pila = [(1, 1)]
    while pila:                                    # excavación iterativa (sin recursión)
        r, c = pila[-1]
        vecinos = [(r + dr * 2, c + dc * 2, r + dr, c + dc) for dr, dc in ORDEN
                   if 0 < r + dr * 2 < H - 1 and 0 < c + dc * 2 < W - 1
                   and g[r + dr * 2, c + dc * 2] == 1]
        if vecinos:
            nr, nc, pr, pc = rng.choice(vecinos)
            g[pr, pc] = 0
            g[nr, nc] = 0
            pila.append((nr, nc))
        else:
            pila.pop()
    if prob_ciclos > 0:                            # derribar paredes -> ciclos
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if g[r, c] == 1 and (r + c) % 2 == 1 and rng.random() < prob_ciclos:
                    g[r, c] = 0
    return g


def laberinto_abierto(celdas_alto: int, celdas_ancho: int) -> np.ndarray:
    """Sala sin paredes interiores (peor caso de optimalidad para DFS)."""
    H, W = 2 * celdas_alto + 1, 2 * celdas_ancho + 1
    g = np.ones((H, W), dtype=np.int8)
    g[1:-1, 1:-1] = 0
    return g


def inicio_meta(g: np.ndarray) -> Tuple[Nodo, Nodo]:
    return (1, 1), (g.shape[0] - 2, g.shape[1] - 2)


# ----------------------------------------------------------------------------
# 2. REPRESENTACIÓN COMO GRAFO (lista de adyacencia + NetworkX)
# ----------------------------------------------------------------------------
def a_lista_adyacencia(g: np.ndarray) -> Dict[Nodo, List[Nodo]]:
    """Generador de estados: sucesores de cada celda libre en orden fijo."""
    adj: Dict[Nodo, List[Nodo]] = {}
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if g[r, c] == 0:
                adj[(r, c)] = [(r + dr, c + dc) for dr, dc in ORDEN
                               if 0 <= r + dr < H and 0 <= c + dc < W and g[r + dr, c + dc] == 0]
    return adj


def a_networkx(adj: Dict[Nodo, List[Nodo]]) -> nx.Graph:
    G = nx.Graph()
    for u, vs in adj.items():
        G.add_node(u)
        for v in vs:
            G.add_edge(u, v)
    return G


# ----------------------------------------------------------------------------
# 3. ALGORITMOS DE BÚSQUEDA CIEGA
# ----------------------------------------------------------------------------
@dataclass
class Resultado:
    nombre: str
    camino: Optional[List[Nodo]]
    expandidos: List[Nodo] = field(default_factory=list)   # orden de expansión
    max_frontera: int = 0
    tiempo_s: float = 0.0
    memoria_kb: float = 0.0

    @property
    def n_expandidos(self) -> int:
        return len(self.expandidos)

    @property
    def longitud(self) -> int:
        return len(self.camino) - 1 if self.camino else -1


def reconstruir(padre: Dict[Nodo, Optional[Nodo]], meta: Nodo) -> List[Nodo]:
    camino, n = [], meta
    while n is not None:
        camino.append(n)
        n = padre[n]
    return camino[::-1]


def bfs(adj, inicio: Nodo, meta: Nodo) -> Resultado:
    """
    BÚSQUEDA EN ANCHURA: frontera = COLA (FIFO). Expande por niveles, por lo que la
    primera vez que llega a la meta lo hace por el camino con menos aristas.
    """
    frontera = deque([inicio])
    padre: Dict[Nodo, Optional[Nodo]] = {inicio: None}   # sirve también como 'visitados'
    res = Resultado("BFS", None, max_frontera=1)
    while frontera:
        n = frontera.popleft()
        res.expandidos.append(n)
        if n == meta:
            res.camino = reconstruir(padre, meta)
            return res
        for v in adj[n]:
            if v not in padre:
                padre[v] = n
                frontera.append(v)
        res.max_frontera = max(res.max_frontera, len(frontera))
    return res


def dfs(adj, inicio: Nodo, meta: Nodo) -> Resultado:
    """
    BÚSQUEDA EN PROFUNDIDAD (iterativa): frontera = PILA (LIFO). Avanza lo más
    hondo posible y retrocede al encontrar un callejón sin salida. Implementada sin
    recursión para evitar el límite de recursión de Python en laberintos grandes.
    """
    pila: List[Tuple[Nodo, Optional[Nodo]]] = [(inicio, None)]
    visitado: Dict[Nodo, Optional[Nodo]] = {}            # nodo -> padre
    res = Resultado("DFS", None, max_frontera=1)
    while pila:
        n, p = pila.pop()
        if n in visitado:
            continue
        visitado[n] = p
        res.expandidos.append(n)
        if n == meta:
            res.camino = reconstruir(visitado, meta)
            return res
        for v in reversed(adj[n]):                      # reversed: se explora primero ARRIBA
            if v not in visitado:
                pila.append((v, n))
        res.max_frontera = max(res.max_frontera, len(pila))
    return res


def medir(algoritmo, adj, inicio, meta, repeticiones: int = 5) -> Resultado:
    """Tiempo = mínimo de varias ejecuciones; memoria pico con tracemalloc."""
    mejor_t = float("inf")
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        res = algoritmo(adj, inicio, meta)
        mejor_t = min(mejor_t, time.perf_counter() - t0)
    tracemalloc.start()
    algoritmo(adj, inicio, meta)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    res.tiempo_s, res.memoria_kb = mejor_t, pico / 1024
    return res


def validar_camino(g: np.ndarray, camino: List[Nodo], inicio: Nodo, meta: Nodo):
    """Comprueba que el camino empieza/termina bien, no cruza paredes y da pasos de 1."""
    assert camino[0] == inicio and camino[-1] == meta
    for (r1, c1), (r2, c2) in zip(camino, camino[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1 and g[r2, c2] == 0


# ----------------------------------------------------------------------------
# 4. EXPERIMENTOS
# ----------------------------------------------------------------------------
def comparar_en(g: np.ndarray, nombre_caso: str):
    inicio, meta = inicio_meta(g)
    adj = a_lista_adyacencia(g)
    G = a_networkx(adj)
    optimo = nx.shortest_path_length(G, inicio, meta)        # verificación independiente
    rb, rd = medir(bfs, adj, inicio, meta), medir(dfs, adj, inicio, meta)
    for r in (rb, rd):
        validar_camino(g, r.camino, inicio, meta)
    assert rb.longitud == optimo, "BFS debe ser óptimo"
    return adj, G, rb, rd, optimo


def fila_tabla(caso, V, E, optimo, r: Resultado):
    return [caso, r.nombre, V, E, optimo, r.longitud, r.n_expandidos,
            round(100 * r.n_expandidos / V, 1), r.max_frontera,
            round(r.tiempo_s * 1000, 3), round(r.memoria_kb, 1)]


ENCABEZADO = ["caso", "algoritmo", "V", "E", "camino_optimo", "camino_hallado",
              "expandidos", "%_explorado", "frontera_max", "tiempo_ms", "memoria_KB"]


def experimento_escalabilidad(lados=(5, 10, 20, 30, 50, 70, 100), semillas=8, p_ciclos=0.10):
    """Promedia sobre varios laberintos con ciclos de tamaño creciente."""
    filas = []
    for lado in lados:
        acum = {"BFS": [], "DFS": []}
        V = 0
        for s in range(semillas):
            g = generar_laberinto(lado, lado, SEMILLA + s, p_ciclos)
            inicio, meta = inicio_meta(g)
            adj = a_lista_adyacencia(g)
            V += len(adj)
            for alg in (bfs, dfs):
                r = medir(alg, adj, inicio, meta, repeticiones=2)
                acum[r.nombre].append((r.n_expandidos, r.tiempo_s * 1000, r.longitud,
                                       r.max_frontera, r.memoria_kb))
        V //= semillas
        for nombre, vals in acum.items():
            m = np.mean(vals, axis=0)
            filas.append((lado, V, nombre, *m))
    return filas


# ----------------------------------------------------------------------------
# 5. GRÁFICAS
# ----------------------------------------------------------------------------
def dibujar_busqueda(ax, g, res: Resultado, inicio, meta, titulo):
    """Celdas coloreadas según el orden de expansión + camino solución en rojo."""
    orden = np.full(g.shape, np.nan)
    for k, (r, c) in enumerate(res.expandidos):
        orden[r, c] = k
    ax.imshow(np.where(g == 1, 0.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5)   # paredes
    ax.imshow(orden, cmap="YlGnBu_r", alpha=.9)
    libres = np.where(g == 0, 1.0, np.nan)
    ax.imshow(libres, cmap="Greys", vmin=0, vmax=3, alpha=.25)                  # no explorado
    ys, xs = zip(*res.camino)
    ax.plot(xs, ys, color="#d62728", lw=2.2)
    ax.scatter(inicio[1], inicio[0], c="#2ca02c", s=90, zorder=5, edgecolors="k")
    ax.scatter(meta[1], meta[0], c="#ff7f0e", s=90, zorder=5, edgecolors="k")
    ax.set_title(f"{titulo}\nexpandidos={res.n_expandidos} · camino={res.longitud} pasos", fontsize=10)
    ax.axis("off")


def graf_laberinto(g, rb, rd, titulo_extra, ruta_salida: Path):
    inicio, meta = inicio_meta(g)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    dibujar_busqueda(ax[0], g, rb, inicio, meta, "BFS (cola FIFO)")
    dibujar_busqueda(ax[1], g, rd, inicio, meta, "DFS (pila LIFO)")
    fig.suptitle(f"{titulo_extra}   |   azul oscuro = expandido primero · gris = no explorado · "
                 "rojo = camino solución", fontsize=10)
    fig.tight_layout(); fig.savefig(ruta_salida, dpi=150)


def graf_grafo_pequeno(ruta_salida: Path):
    """Representación explícita del grafo (nodos = celdas libres) de un laberinto 5x5."""
    g = generar_laberinto(5, 5, SEMILLA, prob_ciclos=0.12)
    inicio, meta = inicio_meta(g)
    adj = a_lista_adyacencia(g); G = a_networkx(adj)
    rb = bfs(adj, inicio, meta)
    pos = {n: (n[1], -n[0]) for n in G.nodes}
    dist = nx.single_source_shortest_path_length(G, inicio)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(g, cmap="gray_r"); ax[0].set_title(f"Laberinto {g.shape[0]}x{g.shape[1]} (negro = pared)")
    ax[0].scatter(inicio[1], inicio[0], c="#2ca02c", s=100, edgecolors="k")
    ax[0].scatter(meta[1], meta[0], c="#ff7f0e", s=100, edgecolors="k"); ax[0].axis("off")
    nodos = nx.draw_networkx_nodes(G, pos, ax=ax[1], node_size=170, node_color=[dist[n] for n in G.nodes],
                                   cmap="viridis_r", edgecolors="k", linewidths=.5)
    nx.draw_networkx_edges(G, pos, ax=ax[1], edge_color="#999")
    nx.draw_networkx_edges(G, pos, ax=ax[1], edgelist=list(zip(rb.camino, rb.camino[1:])),
                           edge_color="#d62728", width=3)
    ax[1].set_title(f"Grafo: |V|={G.number_of_nodes()} nodos, |E|={G.number_of_edges()} aristas\n"
                    "color = distancia al inicio (nivel BFS) · rojo = camino más corto")
    fig.colorbar(nodos, ax=ax[1], shrink=.7, label="nivel BFS"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(ruta_salida, dpi=150)


def graf_escalabilidad(filas, ruta_salida: Path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    series = {"BFS": ("#1f77b4", "o-"), "DFS": ("#d62728", "s--")}
    titulos = [(3, "Nodos expandidos", 0), (4, "Tiempo (ms)", 1), (5, "Longitud del camino (pasos)", 2),
               (6, "Frontera máxima (nodos)", 3)]
    for (idx, titulo, k), a in zip(titulos, ax.ravel()):
        for nombre, (col, est) in series.items():
            pts = [(f[1], f[idx]) for f in filas if f[2] == nombre]
            a.plot(*zip(*pts), est, color=col, label=nombre)
        a.set_title(titulo); a.set_xlabel("|V| (celdas libres)"); a.grid(alpha=.3); a.legend()
    fig.suptitle("Escalabilidad en laberintos con ciclos (promedio de 8 laberintos por tamaño)")
    fig.tight_layout(); fig.savefig(ruta_salida, dpi=150)


# ----------------------------------------------------------------------------
# 6. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    salida = Path(__file__).resolve().parent
    print("=" * 78)
    print("ACTIVIDAD 4 - Búsqueda ciega: BFS vs DFS en laberintos")
    print("=" * 78)

    casos = {
        "Con ciclos 15x15": generar_laberinto(15, 15, SEMILLA, prob_ciclos=0.10),
        "Perfecto 15x15": generar_laberinto(15, 15, SEMILLA, prob_ciclos=0.0),
        "Abierto 15x15": laberinto_abierto(15, 15),
    }
    tabla, guardados = [], {}
    for nombre, g in casos.items():
        adj, G, rb, rd, optimo = comparar_en(g, nombre)
        V, E = G.number_of_nodes(), G.number_of_edges()
        tabla += [fila_tabla(nombre, V, E, optimo, rb), fila_tabla(nombre, V, E, optimo, rd)]
        guardados[nombre] = (g, rb, rd)

    print("\nSUCESORES (generador de estados) del inicio (1,1) en 'Con ciclos 15x15':")
    adj0 = a_lista_adyacencia(casos["Con ciclos 15x15"])
    print("  ", adj0[(1, 1)], "\n")

    print("COMPARACIÓN BFS vs DFS (tiempo = mejor de 5 ejecuciones)")
    cab = f"{'Caso':<18}{'Alg':<5}{'|V|':>6}{'|E|':>6}{'Óptimo':>8}{'Hallado':>9}" \
          f"{'Expand.':>9}{'%Expl.':>8}{'Frontera':>10}{'t (ms)':>9}{'Mem KB':>9}"
    print(cab); print("-" * len(cab))
    for f in tabla:
        print(f"{f[0]:<18}{f[1]:<5}{f[2]:>6}{f[3]:>6}{f[4]:>8}{f[5]:>9}{f[6]:>9}"
              f"{f[7]:>8}{f[8]:>10}{f[9]:>9.3f}{f[10]:>9.1f}")

    # Ejemplo de camino solución (laberinto demostrativo)
    g, rb, rd = guardados["Con ciclos 15x15"]
    print("\nCAMINO SOLUCIÓN BFS (primeras 12 celdas):", rb.camino[:12], "...")
    print("CAMINO SOLUCIÓN DFS (primeras 12 celdas):", rd.camino[:12], "...")

    print("\nESCALABILIDAD (laberintos con ciclos, promedio de 8 por tamaño)")
    filas = experimento_escalabilidad()
    print(f"{'lado':>5}{'|V|':>7}{'Alg':>5}{'Expand.':>10}{'t (ms)':>9}{'Camino':>9}{'Frontera':>10}{'Mem KB':>9}")
    for lado, V, nom, exp, t, cam, fr, mem in filas:
        print(f"{lado:>5}{V:>7}{nom:>5}{exp:>10.0f}{t:>9.2f}{cam:>9.1f}{fr:>10.1f}{mem:>9.0f}")

    # CSV para anexar al repositorio
    with open(salida / "a4_resultados.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(ENCABEZADO); w.writerows(tabla)
        w.writerow([]); w.writerow(["escalabilidad", "lado", "|V|", "algoritmo", "expandidos",
                                    "tiempo_ms", "camino", "frontera_max", "memoria_KB"])
        for lado, V, nom, exp, t, cam, fr, mem in filas:
            w.writerow(["", lado, V, nom, round(exp, 1), round(t, 3), round(cam, 1), round(fr, 1), round(mem, 1)])

    # Gráficas
    graf_grafo_pequeno(salida / "a4_grafo.png")
    for nombre, archivo in (("Con ciclos 15x15", "a4_laberinto_ciclos.png"),
                            ("Perfecto 15x15", "a4_laberinto_perfecto.png"),
                            ("Abierto 15x15", "a4_laberinto_abierto.png")):
        g, rb, rd = guardados[nombre]
        graf_laberinto(g, rb, rd, nombre, salida / archivo)
    graf_escalabilidad(filas, salida / "a4_escalabilidad.png")

    print("""
CONCLUSIONES (contrástalas con los números de las tablas)
  - Optimalidad: BFS siempre devolvió el camino más corto (verificado contra networkx).
    DFS devuelve un camino válido pero NO garantiza el más corto: en el laberinto con
    ciclos halló 112 pasos frente a 100, y en los laberintos grandes su camino llega a
    ser varias veces más largo que el de BFS.
  - Laberinto perfecto (un solo camino posible): ambos devuelven el mismo camino; aquí
    BFS expandió menos nodos que DFS.
  - Nodos expandidos: no hay ganador universal. DFS expande muy pocos si su orden de
    vecinos «acierta» el rumbo (sala abierta: 57 nodos frente a 841 de BFS, con camino
    óptimo por casualidad del orden ARRIBA-DERECHA-ABAJO-IZQUIERDA), pero puede
    explorar casi todo el grafo si falla (laberinto perfecto: 89 %). BFS explora todos los
    nodos más cercanos que la meta, así que en grafos grandes expande casi todo.
  - Memoria: la frontera de BFS crece con el ancho del grafo y la de DFS con su
    profundidad; en nuestros laberintos la pila de DFS (que admite duplicados) llegó a
    tener más entradas, pero la memoria total medida fue menor porque BFS guarda además
    un diccionario de padres de casi todos los nodos.
  - Idoneidad: BFS cuando se necesita el camino de menos pasos (costes uniformes) y el
    grafo cabe en memoria; DFS cuando basta CUALQUIER solución, el espacio es profundo y
    con muchas soluciones, o se quiere ahorrar memoria. Ninguno usa información de la
    meta; por eso en la siguiente actividad se emplean heurísticas (A*).
""")
    print("Archivos: a4_grafo.png, a4_laberinto_ciclos.png, a4_laberinto_perfecto.png,\n"
          "          a4_laberinto_abierto.png, a4_escalabilidad.png, a4_resultados.csv")
    plt.show()


if __name__ == "__main__":
    main()
