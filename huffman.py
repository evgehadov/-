"""
huffman.py — Ядро алгоритма Хаффмана
======================================
Содержит ТОЛЬКО:
  - HuffmanNode          : узел дерева
  - _validate_probabilities : валидация входных данных
  - build_huffman_tree   : построение дерева
  - generate_codes       : генерация бинарных кодов
  - calculate_metrics    : информационные метрики

Не содержит: визуализацию, тесты, UI, точку входа.
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# СТРУКТУРА ДАННЫХ — Узел дерева Хаффмана
# ══════════════════════════════════════════════════════════════════════════════

class HuffmanNode:
    """
    Узел дерева Хаффмана.

    Параметры
    ---------
    symbol : str | None
        Символ алфавита. None для внутренних (не листовых) узлов.
    prob : float
        Вероятность символа или суммарный вес поддерева.
    left : HuffmanNode | None
        Левый потомок — ветвь с кодом «0».
    right : HuffmanNode | None
        Правый потомок — ветвь с кодом «1».

    Исключения
    ----------
    TypeError  — если prob не число или symbol не строка/None
    ValueError — если prob отрицательная
    """

    __slots__ = ("symbol", "prob", "left", "right")

    def __init__(
        self,
        symbol: Optional[str],
        prob: float,
        left:  "Optional[HuffmanNode]" = None,
        right: "Optional[HuffmanNode]" = None,
    ) -> None:
        # Проверка типа вероятности
        if not isinstance(prob, (int, float)):
            raise TypeError(
                f"Вероятность должна быть числом (int/float), "
                f"получено: {type(prob).__name__!r}"
            )
        # Проверка на NaN / Inf прямо здесь — до проверки знака
        if isinstance(prob, float) and (math.isnan(prob) or math.isinf(prob)):
            raise ValueError(
                f"Вероятность не может быть NaN или Inf, получено: {prob}"
            )
        # Проверка знака
        if prob < 0:
            raise ValueError(
                f"Вероятность не может быть отрицательной, получено: {prob}"
            )
        # Проверка символа
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(
                f"Символ должен быть строкой или None, "
                f"получено: {type(symbol).__name__!r}"
            )

        self.symbol: Optional[str]       = symbol
        self.prob:   float               = float(prob)
        self.left:   Optional[HuffmanNode] = left
        self.right:  Optional[HuffmanNode] = right

    # ── Сравнение нужно heapq ────────────────────────────────────────────────

    def __lt__(self, other: "HuffmanNode") -> bool:
        """Узел «меньше» если его вероятность меньше."""
        if not isinstance(other, HuffmanNode):
            return NotImplemented
        return self.prob < other.prob

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HuffmanNode):
            return NotImplemented
        return self.prob == other.prob

    # ── Вспомогательные методы ───────────────────────────────────────────────

    def is_leaf(self) -> bool:
        """True если узел является листом (оба потомка отсутствуют)."""
        return self.left is None and self.right is None

    def __repr__(self) -> str:
        sym = repr(self.symbol) if self.symbol is not None else "<internal>"
        return f"HuffmanNode({sym}, prob={self.prob:.6f})"


# ══════════════════════════════════════════════════════════════════════════════
# ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def _validate_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
    """
    Проверяет словарь вероятностей и возвращает нормализованную копию.

    Что проверяется
    ---------------
    1. Тип аргумента — должен быть dict
    2. Словарь не пустой
    3. Каждый ключ — строка
    4. Каждое значение — число (int/float), не NaN, не Inf
    5. Каждое значение >= 0
    6. Сумма всех значений > 0
    7. Если сумма != 1.0 — автонормализация (с предупреждением)

    Параметры
    ---------
    probabilities : dict
        Сырой словарь {символ: вероятность}.

    Возвращает
    ----------
    dict
        Нормализованный словарь {символ: float}, сумма = 1.0.

    Исключения
    ----------
    TypeError  — неверный тип аргумента или ключей/значений
    ValueError — отрицательные вероятности, нулевая сумма, NaN/Inf
    """

    # 1. Тип аргумента
    if not isinstance(probabilities, dict):
        raise TypeError(
            f"Ожидается словарь dict, получено: {type(probabilities).__name__!r}"
        )

    # 2. Не пустой
    if len(probabilities) == 0:
        raise ValueError("Словарь вероятностей не может быть пустым.")

    cleaned: Dict[str, float] = {}

    for sym, prob in probabilities.items():

        # 3. Ключ — строка
        if not isinstance(sym, str):
            raise TypeError(
                f"Ключ должен быть строкой, получено: {type(sym).__name__!r} ({sym!r})"
            )

        # 4а. Значение — число
        if not isinstance(prob, (int, float)):
            raise TypeError(
                f"Вероятность символа {sym!r} должна быть числом, "
                f"получено: {type(prob).__name__!r} ({prob!r})"
            )

        # 4б. NaN / Inf
        if math.isnan(prob) or math.isinf(prob):
            raise ValueError(
                f"Вероятность символа {sym!r} не может быть NaN или Inf: {prob}"
            )

        # 5. Неотрицательность
        if prob < 0:
            raise ValueError(
                f"Вероятность символа {sym!r} отрицательна: {prob}"
            )

        cleaned[sym] = float(prob)

    # 6. Сумма > 0
    total = sum(cleaned.values())
    if total == 0.0:
        raise ValueError(
            "Сумма всех вероятностей равна нулю — невозможно построить дерево."
        )

    # 7. Нормализация
    if not math.isclose(total, 1.0, rel_tol=1e-9):
        cleaned = {sym: p / total for sym, p in cleaned.items()}

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# АЛГОРИТМ — Построение дерева Хаффмана
# ══════════════════════════════════════════════════════════════════════════════

def build_huffman_tree(probabilities: Dict[str, float]) -> HuffmanNode:
    """
    Строит дерево Хаффмана методом жадного алгоритма.

    Алгоритм (метод Хаффмана, 1952)
    --------------------------------
    1. Для каждого символа создаётся листовой узел.
    2. Все узлы помещаются в минимальную кучу (heapq) по вероятности.
    3. Пока в куче > 1 узла:
       а. Извлекаются два узла с наименьшими вероятностями (left, right).
       б. Создаётся родительский узел с prob = left.prob + right.prob.
       в. Родитель помещается обратно в кучу.
    4. Единственный оставшийся узел — корень дерева.

    Вырожденный случай (1 символ)
    ------------------------------
    Корень — внутренний узел, единственный лист — левый потомок с кодом «0».

    Параметры
    ---------
    probabilities : dict
        {символ: вероятность}

    Возвращает
    ----------
    HuffmanNode
        Корень построенного дерева.

    Исключения
    ----------
    TypeError, ValueError — см. _validate_probabilities
    """
    probs = _validate_probabilities(probabilities)

    # Вырожденный случай: один символ → код «0»
    if len(probs) == 1:
        sym, p = next(iter(probs.items()))
        leaf = HuffmanNode(sym, p)
        root = HuffmanNode(None, p, left=leaf)
        return root

    # Создаём начальную кучу из листовых узлов
    heap: List[HuffmanNode] = [
        HuffmanNode(sym, p) for sym, p in probs.items()
    ]
    heapq.heapify(heap)

    # Строим дерево снизу вверх
    while len(heap) > 1:
        left  = heapq.heappop(heap)  # узел с наименьшей вероятностью
        right = heapq.heappop(heap)  # узел со второй наименьшей вероятностью

        parent = HuffmanNode(
            symbol=None,
            prob=left.prob + right.prob,
            left=left,
            right=right,
        )
        heapq.heappush(heap, parent)

    return heap[0]


# ══════════════════════════════════════════════════════════════════════════════
# АЛГОРИТМ — Генерация кодов
# ══════════════════════════════════════════════════════════════════════════════

def generate_codes(
    node:   Optional[HuffmanNode],
    prefix: str = "",
    codes:  Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Рекурсивно обходит дерево и присваивает бинарные коды листьям.

    Правило кодирования
    -------------------
    Переход к левому  потомку → добавляем «0» к текущему префиксу.
    Переход к правому потомку → добавляем «1» к текущему префиксу.
    Достигнут лист → присваиваем символу накопленный префикс.

    Параметры
    ---------
    node   : HuffmanNode | None  — текущий узел обхода
    prefix : str                 — накопленный бинарный префикс
    codes  : dict | None         — аккумулятор результатов

    Возвращает
    ----------
    dict
        {символ: бинарный_код_строкой}  например {'a': '10', 'b': '0'}
    """
    if codes is None:
        codes = {}

    if node is None:
        return codes

    if node.is_leaf():
        # Вырожденный случай: единственный лист без пути → код «0»
        codes[node.symbol] = prefix if prefix else "0"
    else:
        generate_codes(node.left,  prefix + "0", codes)
        generate_codes(node.right, prefix + "1", codes)

    return codes


# ══════════════════════════════════════════════════════════════════════════════
# МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(
    probabilities: Dict[str, float],
    codes:         Dict[str, str],
) -> Dict[str, float]:
    """
    Вычисляет информационные метрики кодирования Хаффмана.

    Формулы
    -------
    Средняя длина кода : L = Σ p(i) * len(code(i))
    Энтропия Шеннона   : H = -Σ p(i) * log₂(p(i))
    Эффективность      : η = H / L * 100 %
    Избыточность       : R = L - H  (бит)

    Теоретическая гарантия: H ≤ L < H + 1  (теорема Хаффмана)

    Параметры
    ---------
    probabilities : dict  — {символ: вероятность}
    codes         : dict  — {символ: бинарный_код}  из generate_codes()

    Возвращает
    ----------
    dict с ключами: avg_length, entropy, efficiency, redundancy

    Исключения
    ----------
    ValueError — если коды не покрывают все символы из probabilities
    """
    probs = _validate_probabilities(probabilities)

    # Проверяем полноту покрытия кодами
    missing = set(probs.keys()) - set(codes.keys())
    if missing:
        raise ValueError(
            f"Коды отсутствуют для символов: {sorted(missing)}"
        )

    # Средняя длина кода
    avg_length: float = sum(
        probs[sym] * len(codes[sym]) for sym in probs
    )

    # Энтропия Шеннона (пропускаем p=0 чтобы избежать log(0))
    entropy: float = -sum(
        p * math.log2(p) for p in probs.values() if p > 0
    )

    # Эффективность (защита от деления на 0)
    efficiency: float = (entropy / avg_length * 100.0) if avg_length > 0 else 0.0

    # Избыточность
    redundancy: float = avg_length - entropy

    return {
        "avg_length": avg_length,
        "entropy":    entropy,
        "efficiency": efficiency,
        "redundancy": redundancy,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ПУБЛИЧНЫЙ API — удобная обёртка для импорта
# ══════════════════════════════════════════════════════════════════════════════

def encode(probabilities: Dict[str, float]) -> Tuple[HuffmanNode, Dict[str, str], Dict[str, float]]:
    """
    Единая точка входа в алгоритм. Вызывается из visualization.py

    Параметры
    ---------
    probabilities : dict  — {символ: вероятность}

    Возвращает
    ----------
    (root, codes, metrics)
        root    : HuffmanNode          — корень дерева
        codes   : dict[str, str]       — бинарные коды
        metrics : dict[str, float]     — метрики кодирования
    """
    root    = build_huffman_tree(probabilities)
    codes   = generate_codes(root)
    metrics = calculate_metrics(probabilities, codes)
    return root, codes, metrics