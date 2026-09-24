import os
from pathlib import Path

os.environ["DASH_DEBUG"] = "false"
os.environ["DASH_HOT_RELOAD"] = "false"
os.environ["FLASK_DEBUG"] = "0"

import dash_cytoscape as cyto
from ai4cps.dash.dashboard import Dash4CPS

from features.data_plots import DataPlots
from features.prior_knowledge import PriorKnowledge
from features.range_monitoring import RangeMonitoring
from features.autoencoder import AutoencoderAD
from features.case_based import CaseBasedDX
from features.structural_knowledge import StructKnowledgeDX
from features.benchmark_results import BenchmarkResults

cyto.load_extra_layouts()

# Default maximum measurement samples per trace (editable in Data Plots > Configure).
# Set to None, or leave the Configure field blank, to show all samples.
# Fault/parameter transitions and downloaded files are unaffected.
DataPlots.max_plot_points = 5_000

dashboard = Dash4CPS(
    css_overrides=[
        ":root { --selfx-sidebar-width: 16rem; }",
        Path(__file__).parent / "assets" / "benchmark.css",
    ],
    show_reevaluate=False,
)
dashboard.add_system(
    "HSU TwinFlow",
    features=[
        PriorKnowledge,
        DataPlots,
        RangeMonitoring,
        AutoencoderAD,
        CaseBasedDX,
        StructKnowledgeDX,
        BenchmarkResults,
    ],
)

if __name__ == "__main__":
    dashboard.run()
