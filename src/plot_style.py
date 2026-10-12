"""One figure style for the whole project (notebooks 04 and 05), sized to drop straight into slides.

Rules: one idea per figure, no figure title (the slide or the notebook heading gives it),
EPFL red for what we look at, grey for references, text always in ink colours.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RED = '#FF0000'        # EPFL red: Model E, plates, the thing we look at
GREY = '#A8A8A3'       # references and original data
DARK = '#3A3A38'       # second series when both are "ours"
INK, MUTED, GRID = '#1F1F1F', '#6B6B6B', '#E8E8E5'

SOLVENTS = ['MeOH', 'ACN', 'DMSO', 'DCM', 'CHCl3']
SOLVENT_LABELS = {'MeOH': 'MeOH', 'ACN': 'ACN', 'DMSO': 'DMSO', 'DCM': 'DCM', 'CHCl3': 'CHCl$_3$'}
# one fixed, colour-blind-safe colour per solvent (validated palette); names are always written next to the marks
SOLVENT_COLORS = dict(zip(SOLVENTS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))

# figure sizes in inches: full slide width, half slide width, square
WIDE, HALF, SQUARE = (11, 4.4), (6.2, 4.4), (5.4, 5.0)

pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def apply():
    plt.rcParams.update({
        'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 13,
        'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'ytick.left': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 1, 'axes.axisbelow': True,
        'legend.frameon': False, 'lines.linewidth': 2.5, 'lines.markersize': 9,
        'figure.dpi': 100, 'savefig.dpi': 220, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    })


def label_bars(ax, bars, fmt="{:.0%}", inside=False, color=INK, size=12):
    """Write the value on top of each bar (or just inside its top)."""
    for b in bars:
        v = b.get_height()
        if v != v:      # NaN
            continue
        y = v - 0.02 * ax.get_ylim()[1] if inside else v + 0.012 * ax.get_ylim()[1]
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(v), ha='center',
                va='top' if inside else 'bottom', fontsize=size, color=color)


def label_line_ends(ax, x, ys, names, colors=None, min_gap=None, dx=None):
    """Write each series name at the end of its line, pushed apart when they would overlap."""
    lo, hi = ax.get_ylim()
    min_gap = min_gap or 0.055 * (hi - lo)
    dx = dx if dx is not None else 0.02 * (x[-1] - x[0])
    last = None
    for y, name, c in sorted(zip(ys, names, colors or [INK] * len(names)), key=lambda t: t[0]):
        yt = y if last is None else max(y, last + min_gap)
        ax.text(x[-1] + dx, yt, name, va='center', fontsize=12, color=INK)
        last = yt
