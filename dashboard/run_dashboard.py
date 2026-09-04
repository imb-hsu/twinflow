import os

os.environ["DASH_DEBUG"] = "false"
os.environ["DASH_HOT_RELOAD"] = "false"
os.environ["FLASK_DEBUG"] = "0"

import dash_cytoscape as cyto
from selfx.dash.dashboard import SelfXDash

from features.data_plots import DataPlots
from features.prior_knowledge import PriorKnowledge

cyto.load_extra_layouts()

dashboard = SelfXDash(css_overrides=[":root { --selfx-sidebar-width: 16rem; }"], show_reevaluate=False)
dashboard.add_system(
    "HSU TwinFlow",
    features=[PriorKnowledge, DataPlots],
)

if __name__ == "__main__":
    dashboard.run()
