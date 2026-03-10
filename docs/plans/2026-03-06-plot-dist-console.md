# plot_dist Auto Renderer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the public `renderer` argument from `plot_dist(...)` and auto-select console or notebook rendering based on runtime environment while preserving existing plot shaping behavior, with console mode printing only.

**Architecture:** Keep `plot_dist(...)` as the public entrypoint, add a private `_detect_plot_renderer()` helper, and route console rendering through the existing `_plot_dist_console(...)` helper. The renderer decision should be conservative: use Plotly only for real notebook kernels and use console output everywhere else. Console mode should print its formatted text and return `None`.

**Tech Stack:** Python, NumPy, Plotly, pytest-style targeted test execution through the existing `flashml/test.py` module.

---

### Task 1: Add failing auto-detection tests

**Files:**
- Modify: `flashml/test.py`
- Test: `flashml/test.py`

**Step 1: Write the failing test**

Add focused tests that:
- call `plot_dist(...)` without a renderer argument, verify console output in the default pytest environment, and verify the return value is `None`
- patch `_detect_plot_renderer()` to force notebook routing and verify Plotly `Figure.show(...)` is used
- verify the public function signature no longer exposes `renderer`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/mnt/c/Dev/flashml/flashml python3 -m pytest flashml/test.py -k "plot_dist_auto or plot_dist_notebook_route or plot_dist_signature" -q`
Expected: FAIL because the public API still has `renderer` and routing is still manual.

**Step 3: Write minimal implementation**

Do not write implementation yet. Proceed only after observing the expected failure.

**Step 4: Run test to verify it passes**

Run after implementation: `python -m pytest flashml/test.py -k console_plot_dist -q`
Expected: PASS

### Task 2: Implement renderer auto-detection

**Files:**
- Modify: `flashml/main_tools/plot_distribution.py`
- Test: `flashml/test.py`

**Step 1: Write the failing test**

Use Task 1 tests as the failing specification.

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/mnt/c/Dev/flashml/flashml python3 -m pytest flashml/test.py -k "plot_dist_auto or plot_dist_notebook_route or plot_dist_signature" -q`
Expected: FAIL for missing behavior, not for syntax or import errors.

**Step 3: Write minimal implementation**

Implement:
- remove `renderer` from the public signature
- `_detect_plot_renderer()` that returns `"notebook"` only for a real notebook kernel
- routing in `plot_dist(...)` for dict, categorical, and histogram paths based on detected renderer
- keep `_plot_dist_console(...)` as the console implementation while making it print-only
- do not append any Plotly-vs-console summary footer to the printed output
- frame each console line to a fixed width with gray double side borders and a
  matching bottom rule
- place `total=...` inside the console title parentheses and remove it from the
  separate info line

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/mnt/c/Dev/flashml/flashml python3 -m pytest flashml/test.py -k "plot_dist_auto or plot_dist_notebook_route or plot_dist_signature" -q`
Expected: PASS

### Task 3: Check notebook path regression risk

**Files:**
- Modify: `flashml/main_tools/plot_distribution.py`
- Test: `flashml/test.py`

**Step 1: Write the failing test**

Reuse current coverage by keeping notebook behavior untouched; no new notebook assertion is required unless refactor introduces a regression.

**Step 2: Run test to verify it fails**

Not applicable unless a regression is observed during Task 2.

**Step 3: Write minimal implementation**

If needed, restore the previous Plotly path behavior while keeping shared helpers internal.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/mnt/c/Dev/flashml/flashml python3 -m pytest flashml/test.py -k "plot_dist_auto or plot_dist_notebook_route or plot_dist_signature" -q`
Expected: PASS for targeted routing coverage.
