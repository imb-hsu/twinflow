"""Combined SelfX view of TwinFlow prior knowledge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dash_cytoscape as cyto
from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
from selfx.backend import features


DOWNLOAD_BUTTON_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #cbd5e1",
    "borderRadius": "0.35rem",
    "color": "#0f172a",
    "cursor": "pointer",
    "fontWeight": 700,
    "padding": "0.45rem 0.75rem",
}


def _download_json_control(button_id: str, download_id: str) -> html.Div:
    return html.Div(
        [
            html.Button(
                "Download JSON",
                id=button_id,
                n_clicks=0,
                style=DOWNLOAD_BUTTON_STYLE,
            ),
            dcc.Download(id=download_id),
        ],
        style={"display": "inline-block"},
    )


STRUCTURAL_HIERARCHY_PATH = (
    Path(__file__).resolve().parents[2] / "prior_knowledge" / "structural_hierarchy.json"
)


class StructuralHierarchy(features.Feature):
    """Render area, component type, and component identifier hierarchy."""

    abstract = False

    def __init__(
        self,
        tr: Any = None,
        periodic: bool = False,
        fetching: bool = False,
    ) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._structural_hierarchy = self._read_structural_hierarchy()
        self._validate_structural_hierarchy()
        self._metadata = self._structural_hierarchy.get("metadata", {})
        self._areas = self._structural_hierarchy["areas"]
        self._area_names = self._ordered_area_names()

    @staticmethod
    def _read_structural_hierarchy() -> dict[str, Any]:
        with STRUCTURAL_HIERARCHY_PATH.open(encoding="utf-8") as file:
            hierarchy = json.load(file)

        if not isinstance(hierarchy, dict):
            raise ValueError("structural_hierarchy.json must contain an object")
        return hierarchy

    @staticmethod
    def _css_token(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _set_component_ids(self) -> None:
        system_name = self._css_token(self.plant_name or "system")
        prefix = f"{system_name}-structural-hierarchy"
        self._collapse_all_id = f"{prefix}-collapse-all"
        self._expand_all_id = f"{prefix}-expand-all"
        self._download_button_id = f"{prefix}-download-button"
        self._download_id = f"{prefix}-download"
        self._details_node_id = {
            "feature": prefix,
            "detail": ALL,
        }

    def _details_id(self, key: str) -> dict[str, str]:
        return {
            "feature": self._details_node_id["feature"],
            "detail": key,
        }

    def _validate_structural_hierarchy(self) -> None:
        areas = self._structural_hierarchy.get("areas")
        if not isinstance(areas, dict) or not areas:
            raise ValueError("structural_hierarchy.json must contain non-empty areas")

        for area_name, component_types in areas.items():
            if not isinstance(area_name, str) or not isinstance(component_types, dict):
                raise ValueError("Invalid area entry in structural_hierarchy.json")
            if not component_types:
                raise ValueError(f"{area_name} must contain component types")

            for component_type, component_ids in component_types.items():
                if not isinstance(component_type, str):
                    raise ValueError(f"Invalid component type in {area_name}")
                if not isinstance(component_ids, list):
                    raise ValueError(
                        f"Component identifiers for {area_name}/{component_type} "
                        "must be lists"
                    )
                if not all(
                    isinstance(component_id, str) for component_id in component_ids
                ):
                    raise ValueError(
                        f"Invalid component identifier in {area_name}/{component_type}"
                    )

    def _ordered_area_names(self) -> tuple[str, ...]:
        area_order = self._metadata.get("areaOrder")
        if not isinstance(area_order, list):
            return tuple(self._areas)

        ordered_area_names = [
            area_name
            for area_name in area_order
            if isinstance(area_name, str) and area_name in self._areas
        ]
        remaining_area_names = [
            area_name for area_name in self._areas if area_name not in ordered_area_names
        ]
        return tuple(ordered_area_names + remaining_area_names)

    @staticmethod
    def _folder_label(name: str, *statistics: tuple[int, str]) -> html.Span:
        return html.Span(
            [
                html.Span(
                    name,
                    style={"fontFamily": "monospace", "fontWeight": 700},
                ),
                *[
                    html.Span(
                        f"{count} {label}",
                        style={
                            "backgroundColor": "#e0f2fe",
                            "borderRadius": "999px",
                            "color": "#0369a1",
                            "fontSize": "0.75rem",
                            "fontWeight": 700,
                            "marginLeft": "0.5rem",
                            "padding": "0.1rem 0.45rem",
                        },
                    )
                    for count, label in statistics
                ],
            ]
        )

    @staticmethod
    def _component_leaf(component_id: str) -> html.Li:
        return html.Li(
            component_id,
            style={
                "borderLeft": "1px solid #cbd5e1",
                "fontFamily": "monospace",
                "listStyleType": "none",
                "marginLeft": "0.4rem",
                "padding": "0.15rem 0 0.15rem 0.9rem",
            },
        )

    def _component_type_tree(
        self,
        area_name: str,
        component_type: str,
        component_ids: list[str],
    ) -> html.Details:
        return html.Details(
            [
                html.Summary(
                    self._folder_label(
                        component_type,
                        (len(component_ids), "components"),
                    ),
                    style={
                        "cursor": "pointer",
                        "fontWeight": 600,
                        "padding": "0.3rem 0",
                    },
                ),
                html.Ul(
                    [
                        self._component_leaf(component_id)
                        for component_id in component_ids
                    ],
                    style={
                        "margin": "0 0 0.35rem 0.9rem",
                        "padding": 0,
                    },
                ),
            ],
            id=self._details_id(f"component-type:{area_name}:{component_type}"),
            style={
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "0.5rem",
                "marginTop": "0.5rem",
                "padding": "0.35rem 0.65rem",
            },
        )

    def _area_tree(self, area_name: str) -> html.Details:
        component_types = self._areas[area_name]
        component_count = sum(
            len(component_ids) for component_ids in component_types.values()
        )

        return html.Details(
            [
                html.Summary(
                    self._folder_label(
                        area_name,
                        (component_count, "components"),
                        (len(component_types), "component types"),
                    ),
                    style={
                        "cursor": "pointer",
                        "fontSize": "1.1rem",
                        "fontWeight": 700,
                        "padding": "0.35rem 0",
                    },
                ),
                html.Div(
                    [
                        self._component_type_tree(
                            area_name,
                            component_type,
                            component_ids,
                        )
                        for component_type, component_ids in sorted(
                            component_types.items()
                        )
                    ],
                    style={"marginLeft": "1rem"},
                ),
            ],
            id=self._details_id(f"area:{area_name}"),
        )

    def _hierarchy_tree(self) -> html.Div:
        return html.Div(
            [
                self._area_tree(area_name)
                for area_name in self._area_names
            ]
        )

    def perform(self, start: Any, end: Any) -> dict[str, int]:
        """Return static metadata when SelfX requests feature computation."""
        component_types = {
            component_type
            for component_types in self._areas.values()
            for component_type in component_types
        }
        component_count = sum(
            len(component_ids)
            for component_types in self._areas.values()
            for component_ids in component_types.values()
        )

        return {
            "areas": len(self._areas),
            "component_types": len(component_types),
            "components": component_count,
        }

    def is_online(self, role: Any) -> bool:
        """Render static structural hierarchy without time-series analysis."""
        return True

    def icon(self) -> str:
        """Return the Material icon for this feature."""
        return "account_tree"

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        """Build the structural hierarchy layout."""
        return html.Div(
            [
                html.Div(
                    [
                        html.Button(
                            "Expand all",
                            id=self._expand_all_id,
                            n_clicks=0,
                            style={
                                "backgroundColor": "#0f172a",
                                "border": "1px solid #0f172a",
                                "borderRadius": "0.35rem",
                                "color": "#ffffff",
                                "cursor": "pointer",
                                "fontWeight": 700,
                                "padding": "0.45rem 0.75rem",
                            },
                        ),
                        html.Button(
                            "Collapse all",
                            id=self._collapse_all_id,
                            n_clicks=0,
                            style=DOWNLOAD_BUTTON_STYLE,
                        ),
                        _download_json_control(
                            self._download_button_id,
                            self._download_id,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "0.5rem",
                    },
                ),
                html.Div(
                    children=self._hierarchy_tree(),
                    style={
                        "backgroundColor": "#f8fafc",
                        "border": "1px solid #e2e8f0",
                        "borderRadius": "0.5rem",
                        "marginTop": "1.5rem",
                        "overflowX": "auto",
                        "padding": "1rem",
                    },
                ),
            ],
            style={"padding": "1rem"},
        )

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        """Register expand/collapse controls for the structural hierarchy tree."""
        if self._callbacks_registered:
            return

        self._set_component_ids()

        @dash_app.callback(
            Output(self._details_node_id, "open"),
            Input(self._expand_all_id, "n_clicks"),
            Input(self._collapse_all_id, "n_clicks"),
            State(self._details_node_id, "id"),
        )
        def toggle_tree_details(
            expand_clicks: int,
            collapse_clicks: int,
            detail_ids: list[dict[str, str]] | None,
        ) -> list[bool]:
            if not detail_ids:
                return []

            triggered_id = callback_context.triggered_id
            if triggered_id == self._collapse_all_id:
                return [False] * len(detail_ids)
            if triggered_id == self._expand_all_id:
                return [True] * len(detail_ids)
            return [no_update] * len(detail_ids)

        @dash_app.callback(
            Output(self._download_id, "data"),
            Input(self._download_button_id, "n_clicks"),
            prevent_initial_call=True,
        )
        def download_structural_hierarchy(n_clicks: int) -> dict[str, Any]:
            return dcc.send_file(
                str(STRUCTURAL_HIERARCHY_PATH),
                filename="structural_hierarchy.json",
            )

        self._callbacks_registered = True

MATERIAL_FLOW_PATH = Path(__file__).resolve().parents[2] / "prior_knowledge" / "material_flow.json"
STRUCTURAL_HIERARCHY_PATH = (
    Path(__file__).resolve().parents[2] / "prior_knowledge" / "structural_hierarchy.json"
)
TRANSPORTED_ITEM_STYLES = {
    "pallet": {
        "line-color": "#ea580c",
        "target-arrow-color": "#ea580c",
        "line-style": "solid",
    },
    "piece": {
        "line-color": "#2563eb",
        "target-arrow-color": "#2563eb",
        "line-style": "solid",
    },
    "defect": {
        "line-color": "#dc2626",
        "target-arrow-color": "#dc2626",
        "line-style": "solid",
    },
}
COMPONENT_TYPE_COLORS = {
    "AMP": "#a855f7",
    "AMR": "#0d9488",
    "CC": "#64748b",
    "CLT": "#0891b2",
    "PR": "#ca8a04",
    "RB": "#16a34a",
    "RC": "#2563eb",
    "RS": "#db2777",
}


class MaterialFlow(features.Feature):
    """Render the directed material-flow graph from material_flow.json."""

    abstract = False

    def __init__(
        self,
        tr: Any = None,
        periodic: bool = False,
        fetching: bool = False,
    ) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._material_flow = self._read_material_flow()
        (
            self._structural_area_by_component_id,
            self._structural_component_types,
            self._area_order,
        ) = self._read_structural_hierarchy_index()
        self._validate_material_flow()
        self._metadata = self._material_flow.get("metadata", {})
        self._nodes = self._material_flow["nodes"]
        self._edges = self._material_flow["edges"]
        self._area_by_node_id = self._build_area_by_node_id()
        self._area_names = self._ordered_area_names()
        self._stylesheet = self._build_stylesheet()

    @staticmethod
    def _read_material_flow() -> dict[str, Any]:
        with MATERIAL_FLOW_PATH.open(encoding="utf-8-sig") as file:
            material_flow = json.load(file)

        if not isinstance(material_flow, dict):
            raise ValueError("material_flow.json must contain an object")
        return material_flow

    @staticmethod
    def _read_structural_hierarchy_index() -> tuple[dict[str, str], set[str], tuple[str, ...]]:
        with STRUCTURAL_HIERARCHY_PATH.open(encoding="utf-8-sig") as file:
            hierarchy = json.load(file)

        areas = hierarchy.get("areas")
        if not isinstance(areas, dict) or not areas:
            raise ValueError("structural_hierarchy.json must contain non-empty areas")

        area_by_component_id = {}
        component_types = set()
        for area_name, area_component_types in areas.items():
            if not isinstance(area_name, str) or not isinstance(
                area_component_types, dict
            ):
                raise ValueError("Invalid area entry in structural_hierarchy.json")

            for component_type, component_ids in area_component_types.items():
                if not isinstance(component_type, str) or not isinstance(
                    component_ids, list
                ):
                    raise ValueError(
                        f"Invalid component type entry in area {area_name}"
                    )
                component_types.add(component_type)
                for component_id in component_ids:
                    if not isinstance(component_id, str):
                        raise ValueError(
                            f"Invalid component identifier in area {area_name}"
                        )
                    area_by_component_id[component_id] = area_name

        area_order = hierarchy.get("metadata", {}).get("areaOrder", [])
        if not isinstance(area_order, list):
            area_order = []

        ordered_areas = tuple(
            area_name
            for area_name in area_order
            if isinstance(area_name, str) and area_name in areas
        )
        remaining_areas = tuple(
            area_name for area_name in areas if area_name not in ordered_areas
        )
        return area_by_component_id, component_types, ordered_areas + remaining_areas

    @staticmethod
    def _css_token(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _set_component_ids(self) -> None:
        system_name = self._css_token(self.plant_name or "system")
        prefix = f"{system_name}-material-flow"
        self._graph_id = f"{prefix}-graph"
        self._download_button_id = f"{prefix}-download-button"
        self._download_id = f"{prefix}-download"

    def _validate_material_flow(self) -> None:
        nodes = self._material_flow.get("nodes")
        edges = self._material_flow.get("edges")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("material_flow.json must contain non-empty nodes")
        if not isinstance(edges, list) or not edges:
            raise ValueError("material_flow.json must contain non-empty edges")

        node_ids = set()
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("Material-flow nodes must be objects")
            for field in ("id", "type"):
                if not isinstance(node.get(field), str):
                    raise ValueError(f"Material-flow node missing string {field}")
            if node["type"] not in self._structural_component_types:
                raise ValueError(
                    "Material-flow node type must be one of the structural hierarchy "
                    f"component types: {node['id']} has {node['type']}"
                )
            for field in ("is_source", "is_sink"):
                if field in node and not isinstance(node[field], bool):
                    raise ValueError(
                        f"Material-flow node {field} must be a boolean: {node['id']}"
                    )
            shared_with = node.get("shared_with", [])
            if not isinstance(shared_with, list) or not all(
                isinstance(shared_component_id, str)
                for shared_component_id in shared_with
            ):
                raise ValueError(
                    "Material-flow node shared_with must be a list of strings: "
                    f"{node['id']}"
                )
            unknown_shared_components = [
                shared_component_id
                for shared_component_id in shared_with
                if shared_component_id not in self._structural_area_by_component_id
            ]
            if unknown_shared_components:
                raise ValueError(
                    "Material-flow node shared_with contains unknown structural "
                    f"component ids: {node['id']} -> {unknown_shared_components}"
                )
            if node["id"] in node_ids:
                raise ValueError(f"Duplicate material-flow node id: {node['id']}")
            node_ids.add(node["id"])

        for edge in edges:
            if not isinstance(edge, dict):
                raise ValueError("Material-flow edges must be objects")
            for field in ("source", "target", "transportedItem"):
                if not isinstance(edge.get(field), str):
                    raise ValueError(f"Material-flow edge missing string {field}")
            if edge["transportedItem"] not in TRANSPORTED_ITEM_STYLES:
                raise ValueError(
                    "Material-flow edge transportedItem must be pallet, piece, "
                    "or defect"
                )
            if edge["source"] not in node_ids:
                raise ValueError(f"Unknown material-flow edge source: {edge['source']}")
            if edge["target"] not in node_ids:
                raise ValueError(f"Unknown material-flow edge target: {edge['target']}")

    def _build_area_by_node_id(self) -> dict[str, str]:
        area_by_node_id = {}
        edge_neighbors_by_node_id: dict[str, set[str]] = {
            node["id"]: set() for node in self._nodes
        }

        for edge in self._edges:
            edge_neighbors_by_node_id[edge["source"]].add(edge["target"])
            edge_neighbors_by_node_id[edge["target"]].add(edge["source"])

        for node in self._nodes:
            node_id = node["id"]
            area_name = self._structural_area_by_component_id.get(node_id)
            if area_name is not None:
                area_by_node_id[node_id] = area_name
                continue

            neighboring_areas = {
                self._structural_area_by_component_id[neighbor_id]
                for neighbor_id in edge_neighbors_by_node_id[node_id]
                if neighbor_id in self._structural_area_by_component_id
            }
            if len(neighboring_areas) != 1:
                raise ValueError(
                    "Material-flow node must either exist in structural_hierarchy.json "
                    "or connect to exactly one structural area: "
                    f"{node_id} resolved to {sorted(neighboring_areas)}"
                )
            area_by_node_id[node_id] = next(iter(neighboring_areas))

        return area_by_node_id

    def _ordered_area_names(self) -> tuple[str, ...]:
        used_area_names = set(self._area_by_node_id.values())
        ordered_area_names = [
            area_name for area_name in self._area_order if area_name in used_area_names
        ]
        remaining_area_names = sorted(used_area_names.difference(ordered_area_names))
        return tuple(ordered_area_names + remaining_area_names)

    def _build_stylesheet(self) -> list[dict[str, Any]]:
        stylesheet = [
            {
                "selector": "node",
                "style": {
                    "background-color": "#64748b",
                    "color": "#0f172a",
                    "font-size": 12,
                    "height": 30,
                    "label": "data(id)",
                    "shape": "round-rectangle",
                    "text-halign": "center",
                    "text-max-width": 70,
                    "text-valign": "center",
                    "text-wrap": "wrap",
                    "width": 74,
                },
            },
            {
                "selector": "edge",
                "style": {
                    "curve-style": "bezier",
                    "line-color": "#94a3b8",
                    "opacity": 0.65,
                    "target-arrow-color": "#94a3b8",
                    "target-arrow-shape": "triangle",
                    "width": 0.8,
                },
            },
            {
                "selector": "node:selected",
                "style": {
                    "border-width": 0,
                },
            },
            {
                "selector": "edge:selected",
                "style": {
                    "opacity": 1,
                    "width": 2,
                },
            },
            {
                "selector": ".area-order-edge",
                "style": {
                    "line-color": "#000000",
                    "opacity": 0,
                    "target-arrow-shape": "none",
                    "width": 1,
                },
            },
            {
                "selector": ".area-group",
                "style": {
                    "background-color": "#e2e8f0",
                    "background-opacity": 0.22,
                    "border-width": 0,
                    "color": "#334155",
                    "font-size": 14,
                    "font-weight": 700,
                    "label": "data(label)",
                    "padding": "24px",
                    "shape": "round-rectangle",
                    "text-halign": "center",
                    "text-valign": "top",
                },
            },
            {
                "selector": ".boundary-marker",
                "style": {
                    "background-opacity": 0,
                    "border-width": 0,
                    "height": 1,
                    "label": "",
                    "opacity": 0,
                    "width": 1,
                },
            },
            {
                "selector": ".boundary-arrow",
                "style": {
                    "control-point-step-size": 18,
                    "curve-style": "bezier",
                    "line-color": "#0f172a",
                    "line-style": "solid",
                    "opacity": 0.85,
                    "target-arrow-color": "#0f172a",
                    "target-arrow-shape": "triangle",
                    "width": 1.4,
                },
            },
            {
                "selector": ".shared-with-edge",
                "style": {
                    "curve-style": "straight",
                    "line-color": "#475569",
                    "line-style": "dashed",
                    "opacity": 0.75,
                    "target-arrow-shape": "none",
                    "width": 1.2,
                },
            },
        ]

        for transported_item, style in TRANSPORTED_ITEM_STYLES.items():
            stylesheet.append(
                {
                    "selector": f".transported-{transported_item}",
                    "style": style,
                }
            )
        for component_type, color in COMPONENT_TYPE_COLORS.items():
            stylesheet.append(
                {
                    "selector": f".component-type-{self._css_token(component_type)}",
                    "style": {
                        "background-color": color,
                    },
                }
            )
        return stylesheet

    def _elements(self) -> list[dict[str, Any]]:
        elements = [
            {
                "data": {
                    "id": self._area_node_id(area_name),
                    "label": area_name,
                },
                "classes": "area-group",
            }
            for area_name in self._area_names
        ]
        elements.extend(
            {
                "data": {
                    "source": self._area_node_id(source_area),
                    "target": self._area_node_id(target_area),
                },
                "classes": "area-order-edge",
            }
            for source_area, target_area in zip(self._area_names, self._area_names[1:])
        )
        elements.extend(
            {
                "data": {
                    key: value
                    for key, value in {
                        "id": node["id"],
                        "type": node["type"],
                        "parent": self._area_node_id(
                            self._area_by_node_id[node["id"]]
                        ),
                        "is_source": node.get("is_source"),
                        "is_sink": node.get("is_sink"),
                        "shared_with": node.get("shared_with"),
                    }.items()
                    if value is not None
                },
                "classes": f"component-type-{self._css_token(node['type'])}",
            }
            for node in self._nodes
        )
        elements.extend(
            self._boundary_marker_element(node, "source")
            for node in self._nodes
            if node.get("is_source") is True
        )
        elements.extend(
            self._boundary_marker_element(node, "sink")
            for node in self._nodes
            if node.get("is_sink") is True
        )
        elements.extend(
            boundary_edge
            for node in self._nodes
            if node.get("is_source") is True
            for boundary_edge in self._source_boundary_edges(node)
        )
        elements.extend(
            boundary_edge
            for node in self._nodes
            if node.get("is_sink") is True
            for boundary_edge in self._sink_boundary_edges(node)
        )
        elements.extend(
            {
                "data": {
                    "source": edge["source"],
                    "target": edge["target"],
                    "transportedItem": edge["transportedItem"],
                },
                "classes": f"transported-{edge['transportedItem']}",
            }
            for edge in self._edges
        )
        elements.extend(
            self._shared_with_edge(node, shared_component_id)
            for node in self._nodes
            for shared_component_id in node.get("shared_with", [])
        )
        return elements

    @staticmethod
    def _area_node_id(area_name: str) -> str:
        return f"area-{area_name}"

    def _boundary_node_id(self, node_id: str, boundary_type: str) -> str:
        return f"boundary-{boundary_type}-{node_id}"

    def _boundary_marker_element(
        self,
        node: dict[str, Any],
        boundary_type: str,
    ) -> dict[str, Any]:
        return {
            "data": {
                "id": self._boundary_node_id(node["id"], boundary_type),
                "parent": self._area_node_id(self._area_by_node_id[node["id"]]),
            },
            "classes": "boundary-marker",
        }

    def _source_boundary_edges(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            self._boundary_edge(
                source=self._boundary_node_id(node["id"], "source"),
                target=node["id"],
                transported_item=transported_item,
                boundary_type="source",
            )
            for transported_item in self._outgoing_transport_items(node["id"])
        ]

    def _sink_boundary_edges(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            self._boundary_edge(
                source=node["id"],
                target=self._boundary_node_id(node["id"], "sink"),
                transported_item=transported_item,
                boundary_type="sink",
            )
            for transported_item in self._incoming_transport_items(node["id"])
        ]

    def _boundary_edge(
        self,
        source: str,
        target: str,
        transported_item: str,
        boundary_type: str,
    ) -> dict[str, Any]:
        return {
            "data": {
                "id": (
                    f"boundary-{boundary_type}-{source}-{target}-"
                    f"{transported_item}"
                ),
                "source": source,
                "target": target,
                "transportedItem": transported_item,
            },
            "classes": (
                f"boundary-arrow {boundary_type}-boundary-arrow "
                f"transported-{transported_item}"
            ),
        }

    def _incoming_transport_items(self, node_id: str) -> tuple[str, ...]:
        incoming_items = {
            edge["transportedItem"] for edge in self._edges if edge["target"] == node_id
        }
        return self._ordered_transport_items(incoming_items)

    def _outgoing_transport_items(self, node_id: str) -> tuple[str, ...]:
        outgoing_items = {
            edge["transportedItem"] for edge in self._edges if edge["source"] == node_id
        }
        return self._ordered_transport_items(outgoing_items)

    @staticmethod
    def _ordered_transport_items(transported_items: set[str]) -> tuple[str, ...]:
        return tuple(
            transported_item
            for transported_item in TRANSPORTED_ITEM_STYLES
            if transported_item in transported_items
        )

    @staticmethod
    def _shared_with_edge(
        node: dict[str, Any],
        shared_component_id: str,
    ) -> dict[str, Any]:
        return {
            "data": {
                "source": node["id"],
                "target": shared_component_id,
            },
            "classes": "shared-with-edge",
        }

    def perform(self, start: Any, end: Any) -> dict[str, int]:
        """Return static metadata when SelfX requests feature computation."""
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "areas": len(self._area_names),
        }

    def is_online(self, role: Any) -> bool:
        """Render static material-flow knowledge without time-series analysis."""
        return True

    def icon(self) -> str:
        """Return the Material icon for this feature."""
        return "alt_route"

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        """Build the material-flow graph layout."""
        return html.Div(
            [
                html.Div(
                    _download_json_control(
                        self._download_button_id,
                        self._download_id,
                    ),
                    style={"display": "flex", "justifyContent": "flex-end"},
                ),
                html.Div(
                    [
                        cyto.Cytoscape(
                            id=self._graph_id,
                            elements=self._elements(),
                            layout={
                                "name": "klay",
                                "animate": False,
                                "fit": True,
                                "nodeDimensionsIncludeLabels": True,
                                "padding": 40,
                                "klay": {
                                    "borderSpacing": 12,
                                    "aspectRatio": 2.6,
                                    "compactComponents": True,
                                    "crossingMinimization": "LAYER_SWEEP",
                                    "cycleBreaking": "GREEDY",
                                    "direction": "RIGHT",
                                    "edgeRouting": "ORTHOGONAL",
                                    "edgeSpacingFactor": 0.35,
                                    "inLayerSpacingFactor": 0.75,
                                    "layoutHierarchy": True,
                                    "nodeLayering": "NETWORK_SIMPLEX",
                                    "nodePlacement": "LINEAR_SEGMENTS",
                                    "separateConnectedComponents": True,
                                    "spacing": 16,
                                    "thoroughness": 7,
                                },
                            },
                            autoRefreshLayout=False,
                            maxZoom=2.0,
                            minZoom=0.12,
                            stylesheet=self._stylesheet,
                            style={"height": "900px", "width": "100%"},
                            wheelSensitivity=0.15,
                            userPanningEnabled=True,
                            userZoomingEnabled=True,
                        ),
                    ],
                    style={
                        "backgroundColor": "#f8fafc",
                        "border": "1px solid #e2e8f0",
                        "borderRadius": "0.5rem",
                        "padding": "1rem",
                    },
                ),
            ],
            style={"padding": "1rem"},
        )

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        """Register the material-flow download callback."""
        if self._callbacks_registered:
            return

        self._set_component_ids()

        @dash_app.callback(
            Output(self._download_id, "data"),
            Input(self._download_button_id, "n_clicks"),
            prevent_initial_call=True,
        )
        def download_material_flow(n_clicks: int) -> dict[str, Any]:
            return dcc.send_file(
                str(MATERIAL_FLOW_PATH),
                filename="material_flow.json",
            )

        self._callbacks_registered = True

COMPONENT_TYPE_KNOWLEDGE_PATH = (
    Path(__file__).resolve().parents[2] / "prior_knowledge" / "component_type_knowledge.json"
)
PALLET_CONFIGURATIONS_PATH = (
    Path(__file__).resolve().parents[2] / "prior_knowledge" / "pallet_configurations.json"
)


class ComponentTypeKnowledge(features.Feature):
    """Render variable, parameter, and fault definitions by component type."""

    abstract = False

    def __init__(
        self,
        tr: Any = None,
        periodic: bool = False,
        fetching: bool = False,
    ) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._component_knowledge = self._read_component_knowledge()
        self._validate_component_knowledge()
        self._component_types = tuple(sorted(self._component_knowledge))

    @staticmethod
    def _read_component_knowledge() -> dict[str, Any]:
        with COMPONENT_TYPE_KNOWLEDGE_PATH.open(encoding="utf-8") as file:
            knowledge = json.load(file)

        if not isinstance(knowledge, dict):
            raise ValueError("component_type_knowledge.json must contain an object")
        return knowledge

    @staticmethod
    def _css_token(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _set_component_ids(self) -> None:
        system_name = self._css_token(self.plant_name or "system")
        prefix = f"{system_name}-component-knowledge"
        self._component_type_selector_id = f"{prefix}-component-type-selector"
        self._details_id = f"{prefix}-details"
        self._download_button_id = f"{prefix}-download-button"
        self._download_id = f"{prefix}-download"

    def _validate_component_knowledge(self) -> None:
        if not self._component_knowledge:
            raise ValueError("component_type_knowledge.json must not be empty")

        required_definition_sets = (
            "observableVariables",
            "configurableParameters",
            "faultTypes",
        )
        for component_type, knowledge in self._component_knowledge.items():
            if not isinstance(component_type, str) or not isinstance(knowledge, dict):
                raise ValueError("Invalid component type knowledge entry")

            missing_definition_sets = [
                definition_set
                for definition_set in required_definition_sets
                if definition_set not in knowledge
            ]
            if missing_definition_sets:
                raise ValueError(
                    f"{component_type} is missing {missing_definition_sets}"
                )

            definition_sets = [
                knowledge[definition_set]
                for definition_set in required_definition_sets
            ]
            if not all(isinstance(definitions, list) for definitions in definition_sets):
                raise ValueError(
                    f"Variable definitions for {component_type} must be lists"
                )
            if not all(
                isinstance(definition, dict)
                and isinstance(definition.get("name"), str)
                for definitions in definition_sets
                for definition in definitions
            ):
                raise ValueError(f"Invalid variable definition for {component_type}")

            observable_names = {
                variable["name"] for variable in knowledge["observableVariables"]
            }
            configurable_names = {
                variable["name"]
                for variable in knowledge["configurableParameters"]
            }
            overlap = observable_names & configurable_names
            if overlap:
                raise ValueError(
                    f"{component_type} has variables in both observableVariables "
                    f"and configurableParameters: {sorted(overlap)}"
                )

    @staticmethod
    def _definition_constraint(definition: dict[str, Any]) -> str:
        if "min" in definition or "max" in definition:
            return (
                f"{definition.get('min', '-inf')} to "
                f"{definition.get('max', 'inf')}"
            )
        if "values" in definition:
            return ", ".join(str(value) for value in definition["values"])
        return ""

    def _definition_section(
        self,
        title: str,
        definitions: list[dict[str, Any]],
        show_repair_value: bool = False,
    ) -> html.Div:
        if not definitions:
            return html.Div(
                [
                    html.H4(title),
                    html.P("None", style={"color": "#64748b"}),
                ]
            )

        return html.Div(
            [
                html.H4(title),
                html.Table(
                    [
                        html.Colgroup(
                            [
                                html.Col(style={"width": "40%"}),
                                html.Col(style={"width": "20%"}),
                                html.Col(style={"width": "25%"}),
                                html.Col(style={"width": "15%"}),
                            ]
                        ),
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Name"),
                                    html.Th("Type"),
                                    html.Th("Range / values"),
                                    html.Th(
                                        "Repair value" if show_repair_value else ""
                                    ),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(definition["name"]),
                                        html.Td(definition.get("type", "")),
                                        html.Td(
                                            self._definition_constraint(definition)
                                        ),
                                        html.Td(
                                            str(definition.get("repairValue", ""))
                                            if show_repair_value
                                            else "",
                                        ),
                                    ]
                                )
                                for definition in definitions
                            ]
                        ),
                    ],
                    style={
                        "borderCollapse": "collapse",
                        "tableLayout": "fixed",
                        "width": "100%",
                    },
                ),
            ],
            style={"marginTop": "1.5rem"},
        )

    def _component_details(self, component_type: str) -> html.Div:
        knowledge = self._component_knowledge.get(component_type)
        if knowledge is None:
            return html.Div(
                html.P(
                    f"Unknown component type: {component_type}",
                    style={"color": "#b91c1c"},
                )
            )

        return html.Div(
            [
                html.H3(component_type),
                self._definition_section(
                    "Observable variables",
                    knowledge["observableVariables"],
                ),
                self._definition_section(
                    "Configurable parameters",
                    knowledge["configurableParameters"],
                ),
                self._definition_section(
                    "Fault types",
                    knowledge["faultTypes"],
                    show_repair_value=True,
                ),
            ]
        )

    def perform(self, start: Any, end: Any) -> dict[str, int]:
        """Return static metadata when SelfX requests feature computation."""
        return {
            "component_types": len(self._component_types),
            "observable_variables": sum(
                len(knowledge["observableVariables"])
                for knowledge in self._component_knowledge.values()
            ),
            "configurable_parameters": sum(
                len(knowledge["configurableParameters"])
                for knowledge in self._component_knowledge.values()
            ),
            "fault_types": sum(
                len(knowledge["faultTypes"])
                for knowledge in self._component_knowledge.values()
            ),
        }

    def is_online(self, role: Any) -> bool:
        """Render static component knowledge without time-series analysis."""
        return True

    def icon(self) -> str:
        """Return the Material icon for this feature."""
        return "category"

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        """Build the component knowledge layout."""
        initial_component_type = self._component_types[0]

        return html.Div(
            [
                html.Div(
                    _download_json_control(
                        self._download_button_id,
                        self._download_id,
                    ),
                    style={"display": "flex", "justifyContent": "flex-end"},
                ),
                html.Div(
                    [
                        html.Label("Component type"),
                        dcc.Dropdown(
                            id=self._component_type_selector_id,
                            options=[
                                {
                                    "label": component_type,
                                    "value": component_type,
                                }
                                for component_type in self._component_types
                            ],
                            value=initial_component_type,
                            clearable=False,
                        ),
                    ],
                    style={"marginTop": "1.5rem", "maxWidth": "28rem"},
                ),
                html.Div(
                    id=self._details_id,
                    children=self._component_details(initial_component_type),
                    style={
                        "backgroundColor": "#f8fafc",
                        "border": "1px solid #e2e8f0",
                        "borderRadius": "0.5rem",
                        "marginTop": "1.5rem",
                        "overflowX": "auto",
                        "padding": "1rem",
                    },
                ),
            ],
            style={"padding": "1rem"},
        )

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        """Register the component type selector callback."""
        if self._callbacks_registered:
            return

        self._set_component_ids()

        @dash_app.callback(
            Output(self._details_id, "children"),
            Input(self._component_type_selector_id, "value"),
        )
        def show_component_details(component_type: str) -> html.Div:
            return self._component_details(component_type)

        @dash_app.callback(
            Output(self._download_id, "data"),
            Input(self._download_button_id, "n_clicks"),
            prevent_initial_call=True,
        )
        def download_component_type_knowledge(n_clicks: int) -> dict[str, Any]:
            return dcc.send_file(
                str(COMPONENT_TYPE_KNOWLEDGE_PATH),
                filename="component_type_knowledge.json",
            )

        self._callbacks_registered = True


class PalletConfigurations(features.Feature):
    """Render pallet configuration definitions."""

    abstract = False

    _columns = (
        "HeightPosition",
        "LengthPosition",
        "WidthPosition",
        "ProductType",
        "Weight",
        "ProductionRoute",
    )

    def __init__(
        self,
        tr: Any = None,
        periodic: bool = False,
        fetching: bool = False,
    ) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._pallet_configurations = self._read_pallet_configurations()
        self._validate_pallet_configurations()

    @staticmethod
    def _css_token(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _set_component_ids(self) -> None:
        system_name = self._css_token(self.plant_name or "system")
        prefix = f"{system_name}-pallet-configurations"
        self._download_button_id = f"{prefix}-download-button"
        self._download_id = f"{prefix}-download"

    @staticmethod
    def _read_pallet_configurations() -> dict[str, Any]:
        with PALLET_CONFIGURATIONS_PATH.open(encoding="utf-8-sig") as file:
            pallet_configurations = json.load(file)

        if not isinstance(pallet_configurations, dict):
            raise ValueError("pallet_configurations.json must contain an object")
        return pallet_configurations

    def _validate_pallet_configurations(self) -> None:
        if not self._pallet_configurations:
            raise ValueError("pallet_configurations.json must not be empty")

        for configuration_name, configuration in self._pallet_configurations.items():
            if not isinstance(configuration_name, str) or not isinstance(
                configuration,
                dict,
            ):
                raise ValueError("Invalid pallet configuration entry")

            products = configuration.get("PalletWithProducts")
            if not isinstance(products, list):
                raise ValueError(
                    f"{configuration_name} must contain a PalletWithProducts list"
                )
            if not all(isinstance(product, dict) for product in products):
                raise ValueError(
                    f"{configuration_name} contains invalid product entries"
                )

    @staticmethod
    def _statistic(label: str, value: int | str) -> html.Div:
        return html.Div(
            [
                html.Div(str(value), style={"fontSize": "1.5rem", "fontWeight": 700}),
                html.Div(label, style={"color": "#475569", "fontSize": "0.85rem"}),
            ],
            style={
                "backgroundColor": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "borderRadius": "0.5rem",
                "minWidth": "10rem",
                "padding": "0.75rem 1rem",
                "textAlign": "center",
            },
        )

    @staticmethod
    def _product_type_counts(products: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for product in products:
            product_type = str(product.get("ProductType", ""))
            if not product_type:
                product_type = "Unknown"
            counts[product_type] = counts.get(product_type, 0) + 1
        return counts

    @staticmethod
    def _format_counts(counts: dict[str, int]) -> str:
        return ", ".join(
            f"{product_type}: {count}"
            for product_type, count in sorted(counts.items())
        )

    @staticmethod
    def _cell(value: Any) -> html.Td:
        return html.Td(
            str(value) if value not in (None, "") else "-",
            style={
                "borderTop": "1px solid #e2e8f0",
                "padding": "0.45rem 0.6rem",
                "verticalAlign": "top",
            },
        )

    def _configuration_card(
        self,
        configuration_name: str,
        configuration: dict[str, Any],
    ) -> html.Details:
        products = configuration["PalletWithProducts"]
        product_type_counts = self._product_type_counts(products)

        return html.Details(
            [
                html.Summary(
                    [
                        html.Span(
                            configuration_name,
                            style={"fontFamily": "monospace", "fontWeight": 700},
                        ),
                        html.Span(
                            f"{len(products)} products",
                            style={
                                "backgroundColor": "#e0f2fe",
                                "borderRadius": "999px",
                                "color": "#0369a1",
                                "fontSize": "0.75rem",
                                "fontWeight": 700,
                                "marginLeft": "0.5rem",
                                "padding": "0.1rem 0.45rem",
                            },
                        ),
                    ],
                    style={
                        "cursor": "pointer",
                        "fontSize": "1.05rem",
                        "fontWeight": 700,
                        "padding": "0.35rem 0",
                    },
                ),
                html.P(
                    self._format_counts(product_type_counts),
                    style={"color": "#475569", "margin": "0.35rem 0 0.75rem"},
                ),
                html.Div(
                    html.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th(
                                            column,
                                            style={
                                                "backgroundColor": "#f1f5f9",
                                                "borderBottom": "1px solid #cbd5e1",
                                                "padding": "0.45rem 0.6rem",
                                                "textAlign": "left",
                                            },
                                        )
                                        for column in self._columns
                                    ]
                                )
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            self._cell(product.get(column, ""))
                                            for column in self._columns
                                        ]
                                    )
                                    for product in products
                                ]
                            ),
                        ],
                        style={
                            "borderCollapse": "collapse",
                            "tableLayout": "fixed",
                            "width": "100%",
                        },
                    ),
                    style={"overflowX": "auto"},
                ),
            ],
            open=True,
            style={
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "0.5rem",
                "padding": "0.75rem 1rem",
            },
        )

    def perform(self, start: Any, end: Any) -> dict[str, int]:
        """Return static metadata for pallet configurations."""
        products_by_configuration = [
            configuration["PalletWithProducts"]
            for configuration in self._pallet_configurations.values()
        ]
        product_types = {
            product.get("ProductType")
            for products in products_by_configuration
            for product in products
            if product.get("ProductType")
        }
        return {
            "configurations": len(self._pallet_configurations),
            "products": sum(len(products) for products in products_by_configuration),
            "product_types": len(product_types),
        }

    def is_online(self, role: Any) -> bool:
        """Render static pallet configurations without time-series analysis."""
        return True

    def icon(self) -> str:
        """Return the Material icon for this feature."""
        return "inventory_2"

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        """Build the pallet configurations layout."""
        metadata = self.perform(start, end)

        return html.Div(
            [
                html.P("Inspect pallet configurations and their products."),
                html.Div(
                    _download_json_control(
                        self._download_button_id,
                        self._download_id,
                    ),
                    style={"display": "flex", "justifyContent": "flex-end"},
                ),
                html.Div(
                    [
                        self._statistic("configurations", metadata["configurations"]),
                        self._statistic("products", metadata["products"]),
                        self._statistic("product types", metadata["product_types"]),
                    ],
                    style={"display": "flex", "flexWrap": "wrap", "gap": "0.75rem"},
                ),
                html.Div(
                    [
                        self._configuration_card(configuration_name, configuration)
                        for configuration_name, configuration in sorted(
                            self._pallet_configurations.items()
                        )
                    ],
                    style={
                        "display": "grid",
                        "gap": "1rem",
                        "marginTop": "1.5rem",
                    },
                ),
            ],
            style={"padding": "1rem"},
        )

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        """Register the pallet configuration download callback."""
        if self._callbacks_registered:
            return

        self._set_component_ids()

        @dash_app.callback(
            Output(self._download_id, "data"),
            Input(self._download_button_id, "n_clicks"),
            prevent_initial_call=True,
        )
        def download_pallet_configurations(n_clicks: int) -> dict[str, Any]:
            return dcc.send_file(
                str(PALLET_CONFIGURATIONS_PATH),
                filename="pallet_configurations.json",
            )

        self._callbacks_registered = True


class PriorKnowledge(features.Feature):
    """Render TwinFlow prior-knowledge views."""

    abstract = False
    _feature_classes = (
        ("Structural hierarchy", StructuralHierarchy),
        ("Material flow", MaterialFlow),
        ("Component type knowledge", ComponentTypeKnowledge),
        ("Pallet configurations", PalletConfigurations),
    )

    def __init__(
        self,
        tr: Any = None,
        periodic: bool = False,
        fetching: bool = False,
    ) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._features = tuple(
            (label, feature_cls(tr=tr, periodic=periodic, fetching=fetching))
            for label, feature_cls in self._feature_classes
        )

    @staticmethod
    def _css_token(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _set_component_ids(self) -> None:
        system_name = self._css_token(self.plant_name or "system")
        prefix = f"{system_name}-prior-knowledge"
        self._tab_selector_id = f"{prefix}-tab-selector"
        self._tab_content_id = f"{prefix}-tab-content"

    @staticmethod
    def _tab_value(label: str) -> str:
        return label.lower().replace(" ", "-")

    def _sync_child_features(self) -> None:
        self._set_component_ids()
        for _, feature_object in self._features:
            feature_object.plant_name = self.plant_name
            set_component_ids = getattr(feature_object, "_set_component_ids", None)
            if set_component_ids is not None:
                set_component_ids()

    def _feature_by_tab_value(self, tab_value: str) -> features.Feature:
        feature_by_tab_value = {
            self._tab_value(label): feature_object
            for label, feature_object in self._features
        }
        return feature_by_tab_value.get(tab_value, self._features[0][1])

    def _tab_content(
        self,
        tab_value: str,
        role: Any,
        analysis: Any,
        start: Any,
        end: Any,
    ) -> html.Div:
        return self._feature_by_tab_value(tab_value).layout(role, analysis, start, end)

    def perform(self, start: Any, end: Any) -> dict[str, Any]:
        """Return metadata for all prior-knowledge views."""
        self._sync_child_features()
        return {
            label: feature_object.perform(start, end)
            for label, feature_object in self._features
        }

    def is_online(self, role: Any) -> bool:
        """Render static prior knowledge without time-series analysis."""
        return True

    def icon(self) -> str:
        """Return the Material icon for this feature."""
        return "hub"

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        """Build the combined prior-knowledge layout."""
        self._sync_child_features()
        initial_tab = self._tab_value(self._features[0][0])
        return html.Div(
            [
                dcc.Tabs(
                    id=self._tab_selector_id,
                    value=initial_tab,
                    children=[
                        dcc.Tab(label=label, value=self._tab_value(label))
                        for label, _ in self._features
                    ],
                ),
                html.Div(
                    id=self._tab_content_id,
                    children=self._tab_content(initial_tab, role, analysis, start, end),
                ),
            ],
            style={"padding": "1rem"},
        )

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        """Register callbacks for all wrapped prior-knowledge views."""
        if self._callbacks_registered:
            return

        self._sync_child_features()
        for _, feature_object in self._features:
            feature_object.register_callbacks(dash_app, analysis)

        @dash_app.callback(
            Output(self._tab_content_id, "children"),
            Input(self._tab_selector_id, "value"),
        )
        def show_prior_knowledge_tab(tab_value: str) -> html.Div:
            return self._tab_content(tab_value, None, analysis, None, None)

        self._callbacks_registered = True
