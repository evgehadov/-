"""
tests.py — 25 тестов кодирования Хаффмана
==========================================
Импортирует из: huffman.py, visualization.py

Запуск:
    python tests.py

Структура
---------
TestHuffmanNode     (тесты  1- 5) — класс узла
TestValidation      (тесты  6-10) — валидация входных данных
TestBuildTree       (тесты 11-15) — построение дерева
TestGenerateCodes   (тесты 16-18) — генерация кодов
TestMetrics         (тесты 19-20) — информационные метрики
TestVisualization   (тесты 21-23) — визуализация
TestEndToEnd        (тесты 24-25) — полный пайплайн
"""

import math
import sys
import tempfile
import os
import unittest

import matplotlib
matplotlib.use("Agg")  # ← первым делом, до pyplot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from huffman import (
    HuffmanNode,
    build_huffman_tree,
    generate_codes,
    calculate_metrics,
    encode,
)
from visualization import render, save


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ДАННЫЕ — переиспользуются в нескольких тестах
# ══════════════════════════════════════════════════════════════════════════════

PROBS_STANDARD = {
    "а": 0.175, "о": 0.090, "е": 0.072,
    "и": 0.062, "н": 0.053, "т": 0.053,
    "с": 0.045, "р": 0.040, "в": 0.038, "л": 0.035,
}

PROBS_TWO       = {"a": 0.6,   "b": 0.4}
PROBS_EQUAL_4   = {"a": 0.25,  "b": 0.25,  "c": 0.25,  "d": 0.25}
PROBS_EQUAL_8   = {c: 0.125    for c in "abcdefgh"}
PROBS_SKEWED    = {"a": 0.90,  "b": 0.05,  "c": 0.03,  "d": 0.02}
PROBS_ONE       = {"x": 1.0}
PROBS_UNNORM    = {"a": 2,     "b": 3,     "c": 5}      # сумма = 10


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 1 — Класс HuffmanNode (тесты 1-5)
# ══════════════════════════════════════════════════════════════════════════════

class TestHuffmanNode(unittest.TestCase):
    """Проверяем корректность класса HuffmanNode."""

    def test_01_leaf_creation(self):
        """Создание листового узла: symbol и prob сохраняются корректно."""
        node = HuffmanNode("а", 0.175)
        self.assertEqual(node.symbol, "а")
        self.assertAlmostEqual(node.prob, 0.175, places=9)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)

    def test_02_is_leaf_true_and_false(self):
        """is_leaf() True для листа, False для внутреннего узла."""
        leaf   = HuffmanNode("a", 0.5)
        left   = HuffmanNode("b", 0.3)
        right  = HuffmanNode("c", 0.7)
        parent = HuffmanNode(None, 1.0, left=left, right=right)
        self.assertTrue(leaf.is_leaf())
        self.assertFalse(parent.is_leaf())

    def test_03_comparison_for_heapq(self):
        """Узлы сравниваются по prob — необходимо для heapq."""
        n_low  = HuffmanNode("a", 0.1)
        n_high = HuffmanNode("b", 0.9)
        self.assertLess(n_low, n_high)
        self.assertGreater(n_high, n_low)
        self.assertEqual(HuffmanNode("x", 0.5), HuffmanNode("y", 0.5))

    def test_04_negative_prob_raises_value_error(self):
        """Отрицательная вероятность → ValueError."""
        with self.assertRaises(ValueError):
            HuffmanNode("a", -0.01)

    def test_05_non_numeric_prob_raises_type_error(self):
        """Нечисловая вероятность → TypeError."""
        with self.assertRaises(TypeError):
            HuffmanNode("a", "высокая")
        with self.assertRaises(TypeError):
            HuffmanNode("a", [0.5])


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 2 — Валидация входных данных (тесты 6-10)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation(unittest.TestCase):
    """Проверяем что _validate_probabilities отлавливает все некорректные входы."""

    def test_06_empty_dict_raises(self):
        """Пустой словарь → ValueError."""
        with self.assertRaises(ValueError):
            build_huffman_tree({})

    def test_07_not_dict_raises(self):
        """Не словарь (список, строка, None) → TypeError."""
        with self.assertRaises(TypeError):
            build_huffman_tree([("a", 0.5), ("b", 0.5)])
        with self.assertRaises(TypeError):
            build_huffman_tree("ab")
        with self.assertRaises(TypeError):
            build_huffman_tree(None)

    def test_08_negative_probability_raises(self):
        """Отрицательная вероятность любого символа → ValueError."""
        with self.assertRaises(ValueError):
            build_huffman_tree({"a": 0.5, "b": -0.1, "c": 0.6})

    def test_09_nan_and_inf_raise(self):
        """NaN и Inf в вероятностях → ValueError."""
        with self.assertRaises(ValueError):
            build_huffman_tree({"a": float("nan"), "b": 0.5})
        with self.assertRaises(ValueError):
            build_huffman_tree({"a": float("inf"), "b": 0.5})

    def test_10_zero_sum_raises(self):
        """Все вероятности нулевые → ValueError (нечего кодировать)."""
        with self.assertRaises(ValueError):
            build_huffman_tree({"a": 0.0, "b": 0.0, "c": 0.0})


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 3 — Построение дерева (тесты 11-15)
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTree(unittest.TestCase):
    """Проверяем свойства построенного дерева Хаффмана."""

    def test_11_single_symbol_gets_code_zero(self):
        """Один символ → вырожденное дерево, код '0'."""
        root  = build_huffman_tree(PROBS_ONE)
        codes = generate_codes(root)
        self.assertIn("x", codes)
        self.assertEqual(codes["x"], "0")

    def test_12_two_symbols_codes_length_one(self):
        """Два символа → каждый получает код длиной 1 бит."""
        root  = build_huffman_tree(PROBS_TWO)
        codes = generate_codes(root)
        self.assertEqual(len(codes["a"]), 1)
        self.assertEqual(len(codes["b"]), 1)
        self.assertNotEqual(codes["a"], codes["b"])

    def test_13_equal_probs_equal_code_lengths(self):
        """4 равных символа → все коды длиной 2 бита."""
        root  = build_huffman_tree(PROBS_EQUAL_4)
        codes = generate_codes(root)
        lengths = {len(c) for c in codes.values()}
        self.assertEqual(lengths, {2})

    def test_14_unnormalized_probs_accepted(self):
        """Ненормализованные вероятности (сумма≠1) → автонормализация, дерево строится."""
        root  = build_huffman_tree(PROBS_UNNORM)
        codes = generate_codes(root)
        self.assertEqual(set(codes.keys()), {"a", "b", "c"})

    def test_15_root_prob_equals_one(self):
        """Вероятность корня дерева всегда равна 1.0 (после нормализации)."""
        root = build_huffman_tree(PROBS_STANDARD)
        self.assertAlmostEqual(root.prob, 1.0, places=9)


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 4 — Генерация кодов (тесты 16-18)
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateCodes(unittest.TestCase):
    """Проверяем свойства сгенерированных кодов."""

    def test_16_all_symbols_have_codes(self):
        """Каждый символ из входного словаря получает код."""
        root  = build_huffman_tree(PROBS_STANDARD)
        codes = generate_codes(root)
        for sym in PROBS_STANDARD:
            self.assertIn(sym, codes, f"Символ {sym!r} не получил код")

    def test_17_codes_are_binary(self):
        """Все коды состоят исключительно из символов '0' и '1'."""
        root  = build_huffman_tree(PROBS_STANDARD)
        codes = generate_codes(root)
        for sym, code in codes.items():
            self.assertTrue(
                all(bit in "01" for bit in code),
                f"Код символа {sym!r} содержит не бинарные символы: {code!r}",
            )

    def test_18_codes_are_prefix_free(self):
        """
        Коды Хаффмана — префиксные:
        ни один код не является префиксом другого.
        Это гарантирует однозначность декодирования.
        """
        root  = build_huffman_tree(PROBS_STANDARD)
        codes = generate_codes(root)
        code_list = list(codes.values())
        for i, c1 in enumerate(code_list):
            for j, c2 in enumerate(code_list):
                if i != j:
                    self.assertFalse(
                        c2.startswith(c1),
                        f"Код {c1!r} является префиксом {c2!r} — нарушение prefix-free",
                    )


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 5 — Метрики (тесты 19-20)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics(unittest.TestCase):
    """Проверяем корректность информационных метрик."""

    def test_19_entropy_equals_one_for_equal_binary(self):
        """
        Для двух РАВНОВЕРОЯТНЫХ символов энтропия Шеннона = 1.0 бит.
        H = -(0.5·log2(0.5) + 0.5·log2(0.5)) = 1.0
        Используем {"a": 0.5, "b": 0.5} — именно равные, не PROBS_TWO (0.6/0.4).
        """
        root, codes, metrics = encode({"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(metrics["entropy"], 1.0, places=9)

    def test_20_avg_length_geq_entropy(self):
        """
        Теорема Хаффмана: H ≤ L < H + 1.
        Средняя длина кода всегда ≥ энтропии.
        Проверяем на нескольких наборах данных.
        """
        for name, probs in [
            ("standard",  PROBS_STANDARD),
            ("skewed",    PROBS_SKEWED),
            ("equal_8",   PROBS_EQUAL_8),
            ("two",       PROBS_TWO),
        ]:
            with self.subTest(dataset=name):
                _, codes, metrics = encode(probs)
                self.assertGreaterEqual(
                    metrics["avg_length"],
                    metrics["entropy"] - 1e-9,
                    f"L < H для набора {name!r}",
                )
                self.assertLess(
                    metrics["avg_length"],
                    metrics["entropy"] + 1.0 + 1e-9,
                    f"L ≥ H+1 для набора {name!r}",
                )


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 6 — Визуализация (тесты 21-23)
# ══════════════════════════════════════════════════════════════════════════════

class TestVisualization(unittest.TestCase):
    """
    Проверяем что render() и save() работают без исключений.
    Используем setUpClass — рендерим один раз для всего класса.
    """

    @classmethod
    def setUpClass(cls):
        """Один раз строим дерево и рендерим фигуру для всех тестов класса."""
        cls.root, cls.codes, cls.metrics = encode(PROBS_STANDARD)
        cls.probs = PROBS_STANDARD
        cls.fig   = render(cls.root, cls.codes, cls.metrics, cls.probs)

    @classmethod
    def tearDownClass(cls):
        """Закрываем фигуру после всех тестов класса."""
        plt.close(cls.fig)

    def test_21_render_returns_figure(self):
        """render() возвращает объект matplotlib.figure.Figure."""
        self.assertIsInstance(self.fig, Figure)

    def test_22_save_creates_file(self):
        """save() создаёт PNG-файл по указанному пути."""
        root, codes, metrics = encode(PROBS_TWO)
        fig = render(root, codes, metrics, PROBS_TWO)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            save(fig, tmp_path)             # fig закрывается внутри save()
            self.assertTrue(
                os.path.exists(tmp_path),
                f"Файл {tmp_path} не был создан",
            )
            self.assertGreater(
                os.path.getsize(tmp_path), 0,
                "Файл создан, но пустой",
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)         # чистим за собой

    def test_23_render_works_for_edge_cases(self):
        """render() не падает на вырожденных случаях: 1 символ, 2 символа, 8 равных."""
        for name, probs in [
            ("one_symbol", PROBS_ONE),
            ("two_symbols", PROBS_TWO),
            ("equal_8",    PROBS_EQUAL_8),
        ]:
            with self.subTest(case=name):
                root, codes, metrics = encode(probs)
                fig = render(root, codes, metrics, probs)
                try:
                    self.assertIsInstance(fig, Figure)
                finally:
                    plt.close(fig)          # закрываем каждую фигуру


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 7 — Сквозные тесты (тесты 24-25)
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd(unittest.TestCase):
    """Полный пайплайн: вероятности → дерево → коды → метрики → визуализация."""

    def test_24_full_pipeline_russian_alphabet(self):
        """
        Полный пайплайн для подмножества русского алфавита.
        Проверяем: покрытие, бинарность, метрики в диапазоне, Figure создан.
        """
        root, codes, metrics = encode(PROBS_STANDARD)
        fig = render(root, codes, metrics, PROBS_STANDARD)

        try:
            # Все символы покрыты
            self.assertEqual(set(codes.keys()), set(PROBS_STANDARD.keys()))

            # Коды бинарные
            for sym, code in codes.items():
                self.assertTrue(
                    all(b in "01" for b in code),
                    f"Не бинарный код для {sym!r}: {code!r}",
                )

            # Метрики в допустимых диапазонах
            self.assertGreater(metrics["entropy"],    0.0)
            self.assertGreater(metrics["avg_length"], 0.0)
            self.assertGreater(metrics["efficiency"], 50.0)
            self.assertLessEqual(metrics["efficiency"], 100.0 + 1e-9)
            self.assertGreaterEqual(metrics["redundancy"], 0.0)

            # Визуализация создана
            self.assertIsInstance(fig, Figure)

        finally:
            plt.close(fig)

    def test_25_full_pipeline_skewed_probabilities(self):
        """
        Полный пайплайн для резко неравных вероятностей.
        Символ 'a' (p=0.90) должен получить самый короткий код.
        """
        root, codes, metrics = encode(PROBS_SKEWED)
        fig = render(root, codes, metrics, PROBS_SKEWED)

        try:
            # Все символы покрыты
            self.assertEqual(set(codes.keys()), set(PROBS_SKEWED.keys()))

            # Самый частый символ имеет самый короткий (или равный) код
            len_a = len(codes["a"])
            for sym in ["b", "c", "d"]:
                self.assertLessEqual(
                    len_a, len(codes[sym]),
                    f"Код символа 'a' длиннее кода {sym!r}: "
                    f"{codes['a']!r} vs {codes[sym]!r}",
                )

            # Метрики корректны
            self.assertGreater(metrics["entropy"],    0.0)
            self.assertGreater(metrics["avg_length"], 0.0)
            self.assertIsInstance(fig, Figure)

        finally:
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# КРАСИВЫЙ ВЫВОД В КОНСОЛЬ
# ══════════════════════════════════════════════════════════════════════════════

class _PrettyResult(unittest.TextTestResult):
    """
    Кастомный результат: каждый тест выводится с номером, именем и статусом.
    PASS → зелёный  ✓
    FAIL → красный  ✗
    ERROR → жёлтый  !
    """

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self._test_counter = 0

    def startTest(self, test):
        super().startTest(test)
        self._test_counter += 1

    def addSuccess(self, test):
        super().addSuccess(test)
        doc   = test.shortDescription() or test._testMethodName
        label = f"{self.GREEN}✓ PASS{self.RESET}"
        self.stream.write(
            f"  {self.CYAN}[{self._test_counter:02d}]{self.RESET} "
            f"{label}  {doc}\n"
        )
        self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        doc   = test.shortDescription() or test._testMethodName
        label = f"{self.RED}✗ FAIL{self.RESET}"
        self.stream.write(
            f"  {self.CYAN}[{self._test_counter:02d}]{self.RESET} "
            f"{label}  {doc}\n"
        )
        self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        doc   = test.shortDescription() or test._testMethodName
        label = f"{self.YELLOW}! ERROR{self.RESET}"
        self.stream.write(
            f"  {self.CYAN}[{self._test_counter:02d}]{self.RESET} "
            f"{label}  {doc}\n"
        )
        self.stream.flush()


class _PrettyRunner(unittest.TextTestRunner):
    """Подключает _PrettyResult и рисует шапку/итог."""

    resultclass = _PrettyResult

    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    def run(self, test):
        total = test.countTestCases()

        # ── Шапка ────────────────────────────────────────────────────────────
        self.stream.write("\n")
        self.stream.write(
            f"{self.BOLD}{'═' * 62}{self.RESET}\n"
            f"{self.BOLD}  ТЕСТЫ КОДИРОВАНИЯ ХАФФМАНА  "
            f"(всего: {total}){self.RESET}\n"
            f"{self.BOLD}{'═' * 62}{self.RESET}\n\n"
        )

        # Блоки
        blocks = [
            (1,  5,  "Класс HuffmanNode"),
            (6,  10, "Валидация входных данных"),
            (11, 15, "Построение дерева"),
            (16, 18, "Генерация кодов"),
            (19, 20, "Информационные метрики"),
            (21, 23, "Визуализация"),
            (24, 25, "Сквозные тесты (end-to-end)"),
        ]
        for start, end, name in blocks:
            self.stream.write(
                f"{self.CYAN}── {name} "
                f"(тесты {start:02d}–{end:02d}) ──{self.RESET}\n"
            )
            # Запускаем только тесты этого блока
            suite = unittest.TestSuite()
            for t in test:
                for tt in t:
                    num = int(tt._testMethodName.split("_")[1])
                    if start <= num <= end:
                        suite.addTest(tt)
            super().run(suite)
            self.stream.write("\n")

        # ── Итог ─────────────────────────────────────────────────────────────
        result = super().run(test)
        passed = total - len(result.failures) - len(result.errors)
        color  = self.GREEN if not result.failures and not result.errors else self.RED
        self.stream.write(f"{self.BOLD}{'═' * 62}{self.RESET}\n")
        self.stream.write(
            f"{self.BOLD}  ИТОГ: "
            f"{color}{passed}/{total} прошло{self.RESET}\n"
        )
        if result.failures:
            self.stream.write(
                f"  {self.RED}Провалено: {len(result.failures)}{self.RESET}\n"
            )
        if result.errors:
            self.stream.write(
                f"  {self.RED}Ошибок:    {len(result.errors)}{self.RESET}\n"
            )
        self.stream.write(f"{self.BOLD}{'═' * 62}{self.RESET}\n\n")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None   # сохраняем порядок объявления

    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(TestHuffmanNode),
        loader.loadTestsFromTestCase(TestValidation),
        loader.loadTestsFromTestCase(TestBuildTree),
        loader.loadTestsFromTestCase(TestGenerateCodes),
        loader.loadTestsFromTestCase(TestMetrics),
        loader.loadTestsFromTestCase(TestVisualization),
        loader.loadTestsFromTestCase(TestEndToEnd),
    ])

    runner = _PrettyRunner(stream=sys.stdout, verbosity=0)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)