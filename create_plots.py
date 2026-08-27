import json
from ml4cps import vis
from ml4cps.automata import Automaton

# Load material_flow_diagram.json and create Automaton object
with open("material_flow_diagram.json", "r") as f:
    data = json.load(f)

# Convert data to format acceptable by Automaton class
# Assuming the Automaton class expects states, transitions, initial_state, and accepting_states
# automaton_data = {
#     "states": ,
#     "transitions": ,
#     "initial_state": data.get("initial_state", ""),
#     "accepting_states": data.get("accepting_states", []),
# }

automaton = Automaton(states=data.get("nodes", []), transitions=data.get("edges", []))

vis.plot_cps_component(automaton, output="dash", dash_port=8052)
