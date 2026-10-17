"""One visual language for every figure in the project.

seaborn draws the static figures that go into the paper; plotly draws interactive figures
for exploring results. Both use the same colors, so a framing label looks the same in
every figure.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib.figure import Figure

# seaborn's colorblind-safe palette, fixed per framing label.
LABEL_COLORS: dict[str, str] = {
    "eval": "#0173B2",
    "wood": "#DE8F05",
    "deploy": "#029E73",
    "neutral": "#949494",
}
LABEL_ORDER: tuple[str, ...] = tuple(LABEL_COLORS)

PLOTLY_TEMPLATE = "evalgaming"
FONT = "Helvetica, Arial, sans-serif"


def use_style() -> None:
    """Apply the project style to seaborn and register it as plotly's default template."""
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=list(LABEL_COLORS.values()),
        font="sans-serif",
        font_scale=1.2,
    )
    template = go.layout.Template(pio.templates["simple_white"])
    template.layout.colorway = list(LABEL_COLORS.values())
    template.layout.font = {"family": FONT, "size": 14}
    pio.templates[PLOTLY_TEMPLATE] = template
    pio.templates.default = PLOTLY_TEMPLATE


def save(figure: Figure | go.Figure, path: Path) -> list[Path]:
    """Save a figure next to its run: PDF and PNG for seaborn, standalone HTML for plotly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(figure, go.Figure):
        html = path.with_suffix(".html")
        figure.write_html(html, include_plotlyjs="cdn")
        return [html]
    outputs = [path.with_suffix(".pdf"), path.with_suffix(".png")]
    for output in outputs:
        figure.savefig(output, dpi=200, bbox_inches="tight")
    return outputs
