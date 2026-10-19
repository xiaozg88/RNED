"""Shared publication style for run-to-failure figures."""
import matplotlib
import matplotlib.pyplot as plt

FONT_SIZE = 10
FORMAT = "pdf"
DPI = 300
FIG_DIR = "figures"

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
})

# colorblind-safe (Okabe-Ito-like) palette
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def save_fig(fig, name, fmt=FORMAT):
    fig.savefig(f"{FIG_DIR}/{name}.{fmt}")
    fig.savefig(f"{FIG_DIR}/{name}.png")
    print(f"saved {FIG_DIR}/{name}.{{pdf,png}}")
