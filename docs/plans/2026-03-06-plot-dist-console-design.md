# plot_dist Auto Renderer Design

## Summary

Remove the public `renderer` argument from
`flashml.main_tools.plot_distribution.plot_dist`.
The function should auto-detect its environment and choose:
- console rendering for normal Python scripts and terminal-driven execution
- notebook rendering only for a real Jupyter notebook kernel

When console rendering is chosen, the function will render a text-based bar chart
directly in the terminal and print it. Console mode does not return the rendered
string.

## Goals

- Simplify the public `plot_dist(...)` API by removing manual renderer selection.
- Support both categorical/discrete and numeric histogram inputs.
- Preserve the same filtering and shaping behavior already implemented for Plotly:
  dict input, `sort`, `top_n`, `bins`, `title`, `xlabel`, `ylabel`, `bar_color`,
  `size`, `None` buckets, and numeric summary statistics in titles.
- Ignore `xlabel_rotation` and `draw_details` in console mode.
- Default to console outside real notebook kernels to avoid unexpected popups.

## Non-Goals

- No ASCII conversion from images.
- No hover metadata.
- No quantile overlay zones.
- No annotation overlays inside bars.
- No attempt to perfectly mirror Plotly styling.

## Approach

Use a private `_detect_plot_renderer()` helper that returns `"notebook"` only when
running inside a real Jupyter notebook kernel and `"console"` otherwise. Keep the
dedicated `_plot_dist_console(...)` helper for terminal output. The console renderer
will format a terminal-friendly horizontal bar chart with aligned labels, scaled bar
widths, counts, and percentages. For dense plots, labels may wrap into stacked lines
to remain readable without relying on rotated axes. The helper may build the string
internally, but its public effect is printing, not returning.

The printed block should use a fixed-width framed layout so the right edge stays
aligned. Add double vertical side borders with padding on both sides of the content
and color those side margins gray when ANSI output is available. Add a matching
bottom rule line and include `total=...` inside the title parentheses instead of a
separate info field.

## Output Shape

- Title/header line with plot name and summary metadata.
- Optional axis labels when meaningful in text output.
- One row per category/bin with:
  label, bar, raw count, percentage.

## Risks

- Terminal widths vary, so bar sizing must degrade gracefully.
- Very long labels can dominate the output, so truncation or stacked labels is needed.
- Shared prep refactors must not alter existing Plotly output behavior.
- Environment detection must stay conservative so terminal sessions do not get Plotly
  output by mistake.
