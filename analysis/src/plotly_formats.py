#!/usr/bin/env python3
"""
plotly_formats.py

A more "publication ready" Plotly house-style module:
- sensible defaults for thesis printing and PowerPoint
- improved title/tick formatting (sizes, standoff, tick length/width)
- better log-axis tick control to avoid overlapping tick labels
- easy presets (THESIS / PPT / etc.)
- a main() that generates example plots for each preset and permutations

Requires:
  pip install -U kaleido
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Palette
# -----------------------------
PALETTE: Dict[str, str] = {
    "blue_dark": "#4F6F9F",
    "red_main":  "#D30B00",
    "gray_main": "#424548",
    "black":     "#000000",
    "blue_fill": "rgba(79,111,159,0.20)",
}

# -----------------------------
# Default static config
# -----------------------------
STATIC_CONFIG = dict(staticPlot=True, displayModeBar=False)

# -----------------------------
# Specs
# -----------------------------
@dataclass(frozen=True)
class AxisSpec:
    title: str
    title_size: int = 22
    tick_size: int = 16
    tickformat: str = ".3g"
    range: Optional[Tuple[float, float]] = None

    # log axis behaviour (controls overlap)
    log: bool = False
    nticks: Optional[int] = 6  # key to avoid label overlap on log axes

    # axis polish
    title_standoff: int = 12
    ticklen: int = 7
    tickwidth: float = 1.6
    linewidth: float = 1.8

    # spacing between tick labels and axis line (this replaces the invalid tickpadding)
    ticklabelstandoff: int = 8

    showgrid: bool = True
    gridwidth: float = 1.0
    zeroline: bool = False

    # scientific notation
    exponentformat: str = "power"  # nicer in print than 1.0e+3
    showexponent: str = "all"

    # tick label overflow handling
    ticklabeloverflow: str = "hide past domain"
    ticklabelposition: str = "outside"


@dataclass(frozen=True)
class FigureSpec:
    width: int = 900
    height: int = 650

    font_family: str = "Arial, sans-serif"
    font_size: int = 18
    title_size: int = 22

    # margins (generous for export)
    margin_l: int = 95
    margin_r: int = 30
    margin_t: int = 75
    margin_b: int = 110  # helps log tick labels at bottom

    # legend
    legend_x: float = 0.98
    legend_y: float = 0.98
    legend_xanchor: str = "right"
    legend_yanchor: str = "top"
    legend_bg: str = "rgba(255,255,255,0.85)"
    legend_border: str = "rgba(0,0,0,0.15)"
    legend_borderwidth: float = 1.0
    legend_font_size: int = 15

    # axes styling
    axis_linecolor: str = "black"
    axis_showline: bool = True
    axis_mirror: bool = True
    gridcolor: str = "rgba(0,0,0,0.12)"

    # backgrounds
    paper_bgcolor: str = "white"
    plot_bgcolor: str = "white"


# -----------------------------
# Presets
# -----------------------------
PRESETS: Dict[str, FigureSpec] = {
    "THESIS": FigureSpec(
        width=1100, height=780, font_size=20, title_size=24,
        margin_l=110, margin_r=40, margin_t=85, margin_b=125,
        legend_font_size=16
    ),
    "THESIS_SQUARE": FigureSpec(
        width=980, height=980, font_size=20, title_size=24,
        margin_l=115, margin_r=40, margin_t=85, margin_b=135,
        legend_font_size=16
    ),
    "PPT_16_9": FigureSpec(
        width=1280, height=720, font_size=22, title_size=26,
        margin_l=95, margin_r=35, margin_t=70, margin_b=120,
        legend_font_size=18
    ),
    "DRAFT": FigureSpec(
        width=900, height=650, font_size=18, title_size=22,
        margin_l=95, margin_r=30, margin_t=75, margin_b=110,
        legend_font_size=15
    ),
}


# -----------------------------
# Core styling
# -----------------------------
def apply_house_style(
    fig: go.Figure,
    *,
    figspec: FigureSpec = PRESETS["THESIS"],
    title: Optional[str] = None,
    xaxis: Optional[AxisSpec] = None,
    yaxis: Optional[AxisSpec] = None,
    legend: bool = True,
    hover: bool = False,
    template: str = "plotly_white",
) -> go.Figure:
    """Apply consistent 'house style' to a Plotly figure."""

    fig.update_layout(
        template=template,
        width=figspec.width,
        height=figspec.height,
        font=dict(family=figspec.font_family, size=figspec.font_size, color=PALETTE["black"]),
        margin=dict(l=figspec.margin_l, r=figspec.margin_r, t=figspec.margin_t, b=figspec.margin_b),
        paper_bgcolor=figspec.paper_bgcolor,
        plot_bgcolor=figspec.plot_bgcolor,
        hovermode=False if not hover else "closest",
    )

    if title is not None:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5, xanchor="center",
                y=0.98, yanchor="top",
                font=dict(size=figspec.title_size),
            )
        )

    if legend:
        fig.update_layout(
            legend=dict(
                orientation="v",
                x=figspec.legend_x, xanchor=figspec.legend_xanchor,
                y=figspec.legend_y, yanchor=figspec.legend_yanchor,
                bgcolor=figspec.legend_bg,
                bordercolor=figspec.legend_border,
                borderwidth=figspec.legend_borderwidth,
                font=dict(size=figspec.legend_font_size),
            )
        )
    else:
        fig.update_layout(showlegend=False)

    def _axis_kwargs(spec: AxisSpec) -> Dict[str, object]:
        kwargs: Dict[str, object] = dict(
            title=dict(text=spec.title, font=dict(size=spec.title_size), standoff=spec.title_standoff),
            tickfont=dict(size=spec.tick_size),
            tickformat=spec.tickformat,
            range=list(spec.range) if spec.range is not None else None,
            type="log" if spec.log else "linear",
            nticks=spec.nticks,

            ticks="outside",
            ticklen=spec.ticklen,
            tickwidth=spec.tickwidth,
            ticklabelstandoff=spec.ticklabelstandoff,

            showline=figspec.axis_showline,
            linecolor=figspec.axis_linecolor,
            linewidth=spec.linewidth,
            mirror=figspec.axis_mirror,

            showgrid=spec.showgrid,
            gridcolor=figspec.gridcolor,
            gridwidth=spec.gridwidth,
            zeroline=spec.zeroline,

            exponentformat=spec.exponentformat,
            showexponent=spec.showexponent,

            ticklabeloverflow=spec.ticklabeloverflow,
            ticklabelposition=spec.ticklabelposition,

            automargin=True,
        )

        # Keep minor ticks/grid but prevent “busy” visuals.
        # (Doesn't label minors; labels are governed by major ticks.)
        kwargs["minor"] = dict(
            ticklen=max(2, spec.ticklen // 2),
            showgrid=spec.showgrid,
        )

        return kwargs

    if xaxis is not None:
        fig.update_xaxes(**_axis_kwargs(xaxis))
    else:
        fig.update_xaxes(
            ticks="outside",
            showline=figspec.axis_showline,
            linecolor=figspec.axis_linecolor,
            mirror=figspec.axis_mirror,
            gridcolor=figspec.gridcolor,
            automargin=True,
        )

    if yaxis is not None:
        fig.update_yaxes(**_axis_kwargs(yaxis))
    else:
        fig.update_yaxes(
            ticks="outside",
            showline=figspec.axis_showline,
            linecolor=figspec.axis_linecolor,
            mirror=figspec.axis_mirror,
            gridcolor=figspec.gridcolor,
            automargin=True,
        )

    return fig


# -----------------------------
# Common traces
# -----------------------------
def add_envelope_fill(
    fig: go.Figure,
    x_env,
    y_env,
    *,
    name: str,
    fillcolor: str = PALETTE["blue_fill"],
) -> None:
    """Filled polygon band (e.g. mean ± std)."""
    fig.add_trace(go.Scatter(
        x=x_env, y=y_env,
        fill="toself",
        fillcolor=fillcolor,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=name,
    ))

def add_outline(
    fig: go.Figure,
    x, y,
    *,
    color: str = PALETTE["blue_dark"],
    width: float = 2.4,
    opacity: float = 0.7,
    showlegend: bool = False,
    name: str = "",
) -> None:
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(width=width, color=color),
        opacity=opacity,
        showlegend=showlegend,
        hoverinfo="skip",
        name=name if name else None,
    ))

def add_line(
    fig: go.Figure,
    x, y,
    *,
    name: str,
    color: str,
    width: float = 2.4,
    dash: str = "solid",
) -> None:
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(width=width, color=color, dash=dash),
        name=name,
        hoverinfo="skip",
    ))

def add_markers(
    fig: go.Figure,
    x, y,
    *,
    name: str,
    color: str = PALETTE["black"],
    size: int = 7,
    symbol: str = "circle",
) -> None:
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=size, color=color, symbol=symbol),
        name=name,
        hoverinfo="skip",
    ))


# -----------------------------
# Export helpers
# -----------------------------
def show_static(fig: go.Figure) -> None:
    fig.show(config=STATIC_CONFIG)

def save_png(fig: go.Figure, filename: str, *, scale: int = 3) -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(filename, scale=scale)

def save_svg(fig: go.Figure, filename: str) -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(filename)

def save_pdf(fig: go.Figure, filename: str) -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(filename)


# -----------------------------
# Example generator (main)
# -----------------------------
def _example_data(n: int = 60):
    x = np.linspace(0.0, 10.0, n)
    y1 = np.sin(x) + 0.12 * x
    y2 = np.cos(x) + 0.10 * x
    mean = 0.5 * (y1 + y2)
    std = 0.25 + 0.05 * np.sin(0.7 * x)
    lo = mean - std
    hi = mean + std
    return x, y1, y2, mean, lo, hi

def _make_example_figure(*, with_band: bool, with_outline: bool) -> go.Figure:
    x, y1, y2, mean, lo, hi = _example_data()
    fig = go.Figure()

    if with_band:
        x_env = np.concatenate([x, x[::-1]])
        y_env = np.concatenate([hi, lo[::-1]])
        add_envelope_fill(fig, x_env, y_env, name="±1 std")

    if with_outline:
        add_outline(fig, x, y1, name="Outline A", showlegend=False)
        add_outline(fig, x, y2, name="Outline B", showlegend=False)

    add_line(fig, x, mean, name="Mean", color=PALETTE["red_main"], width=3.0)
    add_markers(fig, x[::4], mean[::4], name="Samples", color=PALETTE["gray_main"], size=7)
    return fig

def main():
    """
    Generates example plots for each preset and permutations.

    Output:
      ./plotly_style_examples/<preset>/...
    """
    out_root = Path("plotly_style_examples")
    out_root.mkdir(parents=True, exist_ok=True)

    presets_to_demo = ["THESIS", "THESIS_SQUARE", "PPT_16_9", "DRAFT"]

    permutations = [
        dict(legend=True,  with_band=True,  with_outline=True,  logy=False),
        dict(legend=False, with_band=True,  with_outline=False, logy=False),
        dict(legend=True,  with_band=False, with_outline=True,  logy=False),
        dict(legend=True,  with_band=True,  with_outline=True,  logy=True),
    ]

    for preset_name in presets_to_demo:
        figspec = PRESETS[preset_name]
        preset_dir = out_root / preset_name
        preset_dir.mkdir(parents=True, exist_ok=True)

        for i, perm in enumerate(permutations, start=1):
            if perm["logy"]:
                # log-log example (the one that previously overlapped)
                x = np.linspace(1, 1000, 80)
                y = 1e-3 * (x ** -0.8) + 1e-7
                fig = go.Figure()
                add_line(fig, x, y, name="Power-law decay", color=PALETTE["red_main"], width=3.0)
                xspec = AxisSpec(title="x axis", tickformat=".2g", log=True, nticks=6)
                yspec = AxisSpec(title="y axis", tickformat=".3g", log=True, nticks=6)
            else:
                fig = _make_example_figure(with_band=perm["with_band"], with_outline=perm["with_outline"])
                xspec = AxisSpec(title="x axis", tickformat=".2g", log=False, nticks=None)
                yspec = AxisSpec(title="y axis", tickformat=".3g", log=False, nticks=None)

            apply_house_style(
                fig,
                figspec=figspec,
                title=f"{preset_name} | example {i}",
                xaxis=xspec,
                yaxis=yspec,
                legend=perm["legend"],
                hover=False,
            )

            tag = (
                f"legend{int(perm['legend'])}_"
                f"band{int(perm['with_band'])}_"
                f"outline{int(perm['with_outline'])}_"
                f"logy{int(perm['logy'])}"
            )

            save_png(fig, str(preset_dir / f"example_{i:02d}__{tag}.png"), scale=3)
            save_svg(fig, str(preset_dir / f"example_{i:02d}__{tag}.svg"))

    print(f"[DONE] Wrote examples to: {out_root.resolve()}")

if __name__ == "__main__":
    main()
