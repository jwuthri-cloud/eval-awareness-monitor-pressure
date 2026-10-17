"""Tests for the shared figure style in evalgaming.plots."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns

from evalgaming.plots import LABEL_COLORS, LABEL_ORDER, PLOTLY_TEMPLATE, save, use_style

RATES = pd.DataFrame({"label": list(LABEL_ORDER), "type_hint_rate": [0.8, 0.86, 0.34, 0.5]})


def test_use_style_registers_plotly_template_with_label_colors():
    use_style()
    assert pio.templates.default == PLOTLY_TEMPLATE
    assert list(pio.templates[PLOTLY_TEMPLATE].layout.colorway) == list(LABEL_COLORS.values())


def test_save_seaborn_figure_writes_pdf_and_png(tmp_path):
    use_style()
    grid = sns.catplot(data=RATES, x="label", y="type_hint_rate", kind="bar", order=LABEL_ORDER)
    outputs = save(grid.figure, tmp_path / "figures" / "gap")
    assert [p.suffix for p in outputs] == [".pdf", ".png"]
    assert all(p.stat().st_size > 0 for p in outputs)


def test_save_plotly_figure_writes_html(tmp_path):
    use_style()
    figure = px.bar(RATES, x="label", y="type_hint_rate", color="label")
    (html,) = save(figure, tmp_path / "gap")
    assert html.suffix == ".html"
    assert "plotly" in html.read_text(encoding="utf-8")
