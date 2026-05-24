"""
visualization.py — Визуализация кодирования Хаффмана
=====================================================
Импортирует из: huffman.py
Импортируется из: tests.py

Публичный API
-------------
render(root, codes, metrics, probabilities) -> matplotlib.figure.Figure
save(fig, path)                             -> None
"""

import math
import matplotlib
matplotlib.use("Agg")  # ← до любого импорта pyplot, иначе ошибка на сервере

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

from typing import Dict, List, Optional, Tuple

from huffman import HuffmanNode


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ — обход дерева
# ══════════════════════════════════════════════════════════════════════════════

def _tree_depth(node: Optional[HuffmanNode]) -> int:
    """
    Возвращает глубину дерева.
    Нужна для динамического расчёта размера фигуры и шага dx.

    Защита: если node is None → возвращаем 0 (не падаем).
    """
    if node is None:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _collect_all_nodes(
    node:  Optional[HuffmanNode],
    result: Optional[List[HuffmanNode]] = None,
) -> List[HuffmanNode]:
    """
    Собирает все узлы дерева в плоский список (pre-order обход).
    Используется для отрисовки кружков узлов.
    """
    if result is None:
        result = []
    if node is None:
        return result
    result.append(node)
    _collect_all_nodes(node.left,  result)
    _collect_all_nodes(node.right, result)
    return result


def _compute_layout(
    node:  Optional[HuffmanNode],
    pos:   Dict[int, Tuple[float, float]],
    edges: List[Tuple[int, int, str]],
    x:     float = 0.0,
    depth: int   = 0,
    dx:    float = 1.0,
) -> None:
    """
    Рекурсивно вычисляет координаты (x, y) каждого узла.

    Координаты
    ----------
    x : горизонтальная позиция
    y : -depth  (дерево растёт вниз, корень наверху)

    Параметры
    ---------
    pos   : dict {id(node): (x, y)}    — заполняется in-place
    edges : list [(id_parent, id_child, label)]  — заполняется in-place
    dx    : горизонтальный шаг; делится на 2 при каждом спуске

    Защита: dx не уходит в 0 — минимальный порог 0.05,
    при очень глубоких деревьях узлы не сольются в точку.
    """
    if node is None:
        return

    node_id = id(node)
    pos[node_id] = (x, float(-depth))

    half = max(dx / 2.0, 0.05)  # ← защита от схлопывания

    if node.left is not None:
        edges.append((node_id, id(node.left), "0"))
        _compute_layout(node.left,  pos, edges, x - half, depth + 1, half)

    if node.right is not None:
        edges.append((node_id, id(node.right), "1"))
        _compute_layout(node.right, pos, edges, x + half, depth + 1, half)


# ══════════════════════════════════════════════════════════════════════════════
# ПАНЕЛЬ 1 — Дерево Хаффмана
# ══════════════════════════════════════════════════════════════════════════════

def _draw_tree(
    ax:    plt.Axes,
    root:  HuffmanNode,
    codes: Dict[str, str],
) -> None:
    """
    Рисует дерево Хаффмана на переданной оси ax.

    Цвета
    -----
    Рёбра «0» — зелёные, рёбра «1» — красные/розовые.
    Листовые узлы — синие, внутренние — тёмные.

    Защита
    ------
    - node is None → ранний выход
    - id(node) not in pos → пропускаем (не падаем при неполном дереве)
    - codes.get(sym, '') → не KeyError если символ вдруг отсутствует
    """
    ax.set_facecolor("#1e1e2e")
    ax.set_title("Дерево Хаффмана", color="#cdd6f4", fontsize=13, pad=10)
    ax.axis("off")

    if root is None:
        ax.text(0.5, 0.5, "Дерево пусто", ha="center", va="center",
                color="#f38ba8", transform=ax.transAxes)
        return

    # Вычисляем layout
    pos:   Dict[int, Tuple[float, float]] = {}
    edges: List[Tuple[int, int, str]]     = []
    depth = _tree_depth(root)

    # Начальный dx зависит от глубины — чем глубже, тем шире стартовый шаг
    initial_dx = max(2.0 ** (depth - 2), 1.0)
    _compute_layout(root, pos, edges, x=0.0, depth=0, dx=initial_dx)

    # ── Рёбра ────────────────────────────────────────────────────────────────
    for parent_id, child_id, label in edges:
        if parent_id not in pos or child_id not in pos:
            continue  # защита от неполного pos

        px, py = pos[parent_id]
        cx, cy = pos[child_id]

        color = "#a6e3a1" if label == "0" else "#f38ba8"
        ax.plot([px, cx], [py, cy], color=color, linewidth=1.6, zorder=1)

        # Метка на ребре
        mx, my = (px + cx) / 2, (py + cy) / 2
        ax.text(
            mx, my, label,
            ha="center", va="center", fontsize=8,
            color=color, fontweight="bold", zorder=3,
            bbox=dict(boxstyle="round,pad=0.12", fc="#1e1e2e", ec="none"),
        )

    # ── Узлы ─────────────────────────────────────────────────────────────────
    node_radius = 0.18

    for node in _collect_all_nodes(root):
        nid = id(node)
        if nid not in pos:
            continue

        x, y = pos[nid]

        if node.is_leaf():
            face_color = "#89b4fa"
            code_str   = codes.get(node.symbol, "?")
            label_top  = f"{node.symbol}"
            label_bot  = f"p={node.prob:.3f}\n[{code_str}]"
        else:
            face_color = "#313244"
            label_top  = ""
            label_bot  = f"{node.prob:.3f}"

        circle = plt.Circle(
            (x, y), node_radius,
            color=face_color, zorder=2,
        )
        ax.add_patch(circle)

        # Символ внутри кружка (только для листа)
        if label_top:
            ax.text(x, y, label_top,
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="#1e1e2e", zorder=3)

        # Подпись под кружком
        ax.text(
            x, y - node_radius - 0.06,
            label_bot,
            ha="center", va="top",
            fontsize=7, color="#cdd6f4", zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="#181825",
                      ec="none", alpha=0.85),
        )

    ax.autoscale_view()
    ax.set_aspect("equal")

    # Легенда
    p0 = mpatches.Patch(color="#a6e3a1", label="0 — левая ветвь")
    p1 = mpatches.Patch(color="#f38ba8", label="1 — правая ветвь")
    ax.legend(
        handles=[p0, p1], loc="lower right",
        facecolor="#313244", edgecolor="none",
        labelcolor="#cdd6f4", fontsize=8,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ПАНЕЛЬ 2 — Таблица кодов
# ══════════════════════════════════════════════════════════════════════════════

def _draw_table(
    ax:    plt.Axes,
    probs: Dict[str, float],
    codes: Dict[str, str],
) -> None:
    """
    Рисует таблицу: Символ | Вероятность | Код | Длина кода.
    Строки отсортированы по убыванию вероятности.

    Защита
    ------
    - codes.get(sym, '—') → не KeyError
    - Пустой probs → показываем заглушку
    """
    ax.set_facecolor("#1e1e2e")
    ax.axis("off")
    ax.set_title("Таблица кодов", color="#cdd6f4", fontsize=13, pad=10)

    if not probs:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                color="#f38ba8", transform=ax.transAxes)
        return

    sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    col_labels = ["Символ", "Вероятность", "Код Хаффмана", "Длина (бит)"]
    cell_data  = []

    for sym, p in sorted_items:
        code   = codes.get(sym, "—")
        length = str(len(code)) if code != "—" else "—"
        cell_data.append([sym, f"{p:.4f}", code, length])

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Динамический масштаб — чем больше строк, тем меньше высота ячейки
    row_height = max(1.2, min(2.0, 20.0 / max(len(cell_data), 1)))
    tbl.scale(1.15, row_height)

    header_bg  = "#313244"
    row_bgs    = ["#181825", "#1e1e2e"]

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#45475a")
        if row == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="#89b4fa", fontweight="bold")
        else:
            cell.set_facecolor(row_bgs[row % 2])
            cell.set_text_props(color="#cdd6f4")


# ══════════════════════════════════════════════════════════════════════════════
# ПАНЕЛЬ 3 — Гистограмма длин кодов + метрики
# ══════════════════════════════════════════════════════════════════════════════

def _draw_histogram(
    ax:      plt.Axes,
    probs:   Dict[str, float],
    codes:   Dict[str, str],
    metrics: Dict[str, float],
) -> None:
    """
    Рисует столбчатую диаграмму длин кодов для каждого символа.
    Справа — текстовый блок с метриками.

    Защита
    ------
    - Пустые probs/codes → заглушка
    - metrics.get(..., 0) → не KeyError если метрика отсутствует
    """
    ax.set_facecolor("#1e1e2e")
    ax.set_title("Длины кодов", color="#cdd6f4", fontsize=13, pad=10)

    if not probs or not codes:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center",
                color="#f38ba8", transform=ax.transAxes)
        return

    # Сортируем по убыванию вероятности — тот же порядок что в таблице
    sorted_syms = sorted(probs.keys(), key=lambda s: probs[s], reverse=True)
    lengths     = [len(codes.get(s, "")) for s in sorted_syms]
    x_indices   = range(len(sorted_syms))

    bars = ax.bar(
        x_indices, lengths,
        color="#89b4fa", edgecolor="#45475a", linewidth=0.7,
    )

    # Подписи длин над столбцами
    for bar, length in zip(bars, lengths):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            str(length),
            ha="center", va="bottom",
            color="#cdd6f4", fontsize=8,
        )

    ax.set_xticks(list(x_indices))
    ax.set_xticklabels(sorted_syms, color="#cdd6f4", fontsize=9)
    ax.set_ylabel("Длина кода (бит)", color="#cdd6f4", fontsize=9)
    ax.tick_params(axis="both", colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_color("#45475a")

    # Линия средней длины
    avg = metrics.get("avg_length", 0)
    if avg > 0:
        ax.axhline(
            avg, color="#f38ba8", linewidth=1.4, linestyle="--",
            label=f"Средняя длина: {avg:.3f} бит",
        )
        ax.legend(
            facecolor="#313244", edgecolor="none",
            labelcolor="#cdd6f4", fontsize=8,
        )

    # Блок метрик справа от гистограммы
    H = metrics.get("entropy",    0.0)
    L = metrics.get("avg_length", 0.0)
    n = metrics.get("efficiency", 0.0)
    R = metrics.get("redundancy", 0.0)

    metrics_text = (
        f"  Энтропия        H = {H:.4f} бит\n"
        f"  Средняя длина   L = {L:.4f} бит\n"
        f"  Эффективность   η = {n:.2f} %\n"
        f"  Избыточность    R = {R:.4f} бит"
    )
    ax.text(
        1.03, 0.97,
        metrics_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.5, color="#cdd6f4",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="#313244", ec="#45475a",
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ПУБЛИЧНЫЙ API
# ══════════════════════════════════════════════════════════════════════════════

def render(
    root:    HuffmanNode,
    codes:   Dict[str, str],
    metrics: Dict[str, float],
    probs:   Dict[str, float],
) -> Figure:
    """
    Собирает все три панели в одну фигуру и возвращает её.

    Используется в:
        tests.py → render(...) должен вернуть Figure без исключений

    Параметры
    ---------
    root    : HuffmanNode       — корень дерева (из huffman.encode)
    codes   : dict[str, str]    — бинарные коды (из huffman.encode)
    metrics : dict[str, float]  — метрики (из huffman.encode)
    probs   : dict[str, float]  — исходные вероятности

    Возвращает
    ----------
    matplotlib.figure.Figure
        Фигура НЕ закрывается — закрывает вызывающий код.

    Защита
    ------
    - Динамический figsize: ширина растёт с числом символов
    - Все внутренние функции защищены от None и пустых данных
    """
    if root is None:
        raise ValueError("root не может быть None — передайте результат huffman.encode()")
    if not isinstance(codes, dict) or not codes:
        raise ValueError("codes должен быть непустым словарём")
    if not isinstance(metrics, dict):
        raise ValueError("metrics должен быть словарём")
    if not isinstance(probs, dict) or not probs:
        raise ValueError("probs должен быть непустым словарём")

    n_symbols = len(probs)

    # Динамический размер фигуры
    fig_w = max(18, n_symbols * 1.4)
    fig_h = 14

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1e1e2e")
    fig.suptitle(
        "Кодирование Хаффмана",
        fontsize=20, fontweight="bold",
        color="#cdd6f4", y=0.98,
    )

    gs = GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.55,
        left=0.04, right=0.94,
        top=0.93, bottom=0.05,
    )

    ax_tree = fig.add_subplot(gs[0, :])    # верхняя строка — дерево
    ax_tbl  = fig.add_subplot(gs[1, 0])   # нижний левый — таблица
    ax_hist = fig.add_subplot(gs[1, 1])   # нижний правый — гистограмма

    _draw_tree(ax_tree, root, codes)
    _draw_table(ax_tbl,  probs, codes)
    _draw_histogram(ax_hist, probs, codes, metrics)

    return fig


def save(fig: Figure, path: str = "huffman_result.png") -> None:
    """
    Сохраняет фигуру в PNG и закрывает её (освобождает память).

    Параметры
    ---------
    fig  : Figure — фигура из render()
    path : str    — путь для сохранения

    Защита
    ------
    - Проверяем что fig не None
    - plt.close(fig) вызывается всегда (даже если savefig упал)
    """
    if fig is None:
        raise ValueError("fig не может быть None")

    try:
        fig.savefig(
            path, dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print(f"[✓] Сохранено → {path}")
    finally:
        plt.close(fig)  # ← освобождаем память в любом случае