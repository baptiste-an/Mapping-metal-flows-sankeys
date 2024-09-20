import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import pyarrow.feather as feather
from collections import Counter
import pickle
import pathlib
from gloria_preprocessing_for_sankey import *
import lzma

"""This script contains the functions required to process the data into a format appropriate for GoSankey, and then to plot the Sankey diagram using Plotly. 
The script also contains a function to save the Sankey diagrams as SVG files."""

DATA_PATH = pathlib.Path(__file__).parent


def indexes(region):
    cba_sect_ind = [
        "Building",
        "Civil engineering",
        "Electronic and electrical equipment",
        "Machinery and equipment",
        "Transport equipment",
        "Other manufactured goods",
        "Raw materials",
        "Other",
        "RoW - Building",
        "RoW - Civil engineering",
        "RoW - Electronic and electrical equipment",
        "RoW - Machinery and equipment",
        "RoW - Transport equipment",
        "RoW - Other manufactured goods",
        "RoW - Raw materials",
        "RoW - Other",
    ]
    imp_reg_ind = [
        region + " ",
        "Africa ",
        "Asia-Pacific ",
        "EECCA ",
        "Europe ",
        "Latin America ",
        "Middle East ",
        "North America ",
    ]
    reg_own_ind = [
        region + "  ",
        "Africa  ",
        "Asia-Pacific  ",
        "EECCA  ",
        "Europe  ",
        "Latin America  ",
        "Middle East  ",
        "North America  ",
        "Unknown  ",
    ]
    imp_dom_ind = [region + "    ", "Imports"]
    cba_fd_ind = [
        "Households",
        "Government",
        "NPISHS",
        "GFCF",
        "Acquisitions less disposals of valuables",
        "Inventories",
        "RoW - Households",
        "RoW - Government",
        # "RoW - NPISHS",
        "RoW - GFCF",
        # "RoW - Acquisitions less disposals of valuables",
        # "RoW - Inventories",
        "RoW - Others",
    ]
    cba_reg_ind = [
        region + "   ",
        "Africa   ",
        "Asia-Pacific   ",
        "EECCA   ",
        "Europe   ",
        "Latin America   ",
        "Middle East   ",
        "North America   ",
    ]

    return (
        cba_sect_ind,
        imp_reg_ind,
        reg_own_ind,
        imp_dom_ind,
        cba_fd_ind,
        cba_reg_ind,
    )


def node_y(nodes, node, color, region):
    """This allows to set the y height of the nodes."""

    if type(nodes["position"].loc[node]) == str:
        pos = nodes["position"].loc[node]
    else:
        pos = nodes["position"].loc[node].unique()[0]
    df = nodes.reset_index().set_index(["position", "index"]).loc[pos]["value t"]

    (
        cba_sect_ind,
        imp_reg_ind,
        reg_own_ind,
        imp_dom_ind,
        cba_fd_ind,
        cba_reg_ind,
    ) = indexes(region)
    if pos[3:] == "cba sect":
        df = pd.DataFrame(df, index=cba_sect_ind)["value t"].dropna()
    elif pos[3:] == "region owner":
        df = pd.DataFrame(df, index=reg_own_ind)["value t"].dropna()
    elif pos[3:] == "imp dom":
        df = pd.DataFrame(df, index=imp_dom_ind)["value t"].dropna()
    elif pos[3:] == "cba fd":
        df = pd.DataFrame(df, index=cba_fd_ind)["value t"].dropna()
    elif pos[3:] == "cba reg":
        df = pd.DataFrame(df, index=cba_reg_ind)["value t"].dropna()
    elif pos[3:] == "imp reg":
        df = pd.DataFrame(df, index=imp_reg_ind)["value t"].dropna()

    total = df.sum()
    white = 1 - color

    return (
        len(df.loc[:node]) / (len(df) + 1) * white
        + (df.loc[:node][:-1].sum() + df.loc[node] / 2) / total * color
    )


def Nodes(region, year, height, top_margin, bottom_margin, ratio, position, save_path):
    """Preprocess all the nodes."""

    nodes_path = save_path + "/" + region + "/nodes" + region + str(year) + ".feather"

    nodes = feather.read_feather(nodes_path)

    size = height - top_margin - bottom_margin
    n = max(Counter(nodes["position"].values).values())
    pad = 0.25 * size / (n + 1)  # that's the white when ratio=1 (check to confirm)
    # change this to have more white when ratio=1
    pad2 = (size - ratio * (size - (n - 1) * pad)) / (n + 1)
    # because: ratio = (size - (n+1)*pad2 )/(size - (n - 1) * pad)
    # because with pad2 white space at top and bottom
    white = ((n + 1) * pad2) / size
    color = 1 - white

    positions = {
        item: 0.00001 if i == 0 else i / (len(position) - 1)
        for i, item in enumerate(position)
    }
    nodes["x"] = nodes["position"].map(positions)
    nodes["y"] = nodes.index.map(lambda i: node_y(nodes, i, color, region))

    return nodes, pad, pad2, n


####################################


def preprocess_arrows_and_labels(
    sankey_type,
    y=1,
    yt=1.06,
    arrow_head_length=0.07,
    arrow_head_width=0.09,
    text_size=12.5,
):
    """Add arrows and labels to the Sankey diagram."""
    arrow_y = y  # y-coordinate for arrows
    arrow_text_y = yt  # y-coordinate for text above arrows

    if sankey_type == "Commodity ownership":
        arrow_info = [
            {
                "x0": 0.0,
                "x1": 0.05,
                "texts": ["                         Nationality of mine owners"],
            },
            {
                "x0": 0.18,
                "x1": 0.42,
                "texts": ["Production-based account"],
            },  # Region of mines
            {
                "x0": 0.58,
                "x1": 1.002,
                "texts": ["Consumption-based account"],
            },  # Region of final consumption
        ]
    elif sankey_type == "Commodity":
        arrow_info = [
            {
                "x0": 0.0,
                "x1": 0.05,
                "texts": ["                        Production-based account"],
            },
            {"x0": 0.31, "x1": 1.002, "texts": ["Consumption-based account"]},
        ]

    elif sankey_type == "All commodities":
        arrow_info = [
            {
                "x0": 0.0,
                "x1": 0.05,
                "texts": ["   Commodity"],
            },
            {
                "x0": 0.23,
                "x1": 0.27,
                "texts": ["Production-based account"],
            },  # Region of mines
            {
                "x0": 0.48,
                "x1": 1.002,
                "texts": ["Consumption-based account"],
            },  # Region of final consumption
        ]
    elif sankey_type == "Commodity ownership pba":
        arrow_info = [
            {
                "x0": 0.0,
                "x1": 0.05,
                "texts": ["                   Region of mine owners"],
            },
            {"x0": 0.2, "x1": 0.3, "texts": ["Region of mines"]},
            {"x0": 0.45, "x1": 0.55, "texts": ["Region of ores consumption"]},
            {
                "x0": 0.7,
                "x1": 1.002,
                "texts": ["Region of metal intermediate consumption"],
            },
        ]
    elif sankey_type == "Commodity all ownership":
        arrow_info = [
            {
                "x0": 0.0,
                "x1": 0.05,
                "texts": ["                         Nationality of mine owners"],
            },
            {
                "x0": 0.2,
                "x1": 0.3,
                "texts": ["Production-based account"],
            },  # Region of mines
            {
                "x0": 0.48,
                "x1": 1.002,
                "texts": ["Consumption-based account"],
            },  # Region of final consumption
        ]
    else:
        return {"shapes": [], "annotations": []}

    shapes = []
    annotations = []

    spacing_between_lines = 0.033  # Adjust this for gap between lines
    for arrow in arrow_info:
        # Calculate arrowhead and body points
        left_x = arrow["x0"] + 0.007  # arrowhead_length
        left_y1 = arrow_y - 0.009  # arrowhead_width
        left_y2 = arrow_y + 0.009

        right_x = arrow["x1"] - 0.007
        right_y1 = arrow_y - 0.009
        right_y2 = arrow_y + 0.009

        # Create the shape path
        path = (
            f"M {left_x},{left_y1} L {arrow['x0']},{arrow_y} L {left_x},{left_y2} "
            f"M {arrow['x0']},{arrow_y} L {arrow['x1']},{arrow_y} "
            f"M {right_x},{right_y1} L {arrow['x1']},{arrow_y} L {right_x},{right_y2}"
        )

        # Add the shape to the list
        shapes.append(
            dict(
                type="path",
                path=path,
                xref="paper",
                yref="paper",
                line=dict(color="black", width=2.5),
            )
        )

        # Store annotations
        for i, text in enumerate(arrow["texts"]):
            annotations.append(
                dict(
                    text=text,
                    x=(arrow["x0"] + arrow["x1"]) / 2,
                    y=arrow_text_y - i * spacing_between_lines,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    showarrow=False,
                    font=dict(size=text_size),
                )
            )

    return {"shapes": shapes, "annotations": annotations}


def preprocess_sankey_data(year, region, sankey_type, sankey_subtype=None):
    """Preprocesses and saves all data required to build the Sankey diagram."""

    # Your custom logic to define paths and retrieve data
    position, save_path = variables1(sankey_type)

    # read norm
    # ratio = feather.read_feather(save_path + "/norm.feather").loc[region,year] # ends up not being proportional due to some internal plotly setting
    ratio = 0.9

    # Define paths
    data_sankey_path = (
        save_path + "/" + region + "/data" + region + str(year) + ".feather"
    )
    node_list_path = (
        save_path + "/" + region + "/nodelist" + region + str(year) + ".feather"
    )

    try:
        # Load data from Feather files
        data_sankey = pd.read_feather(data_sankey_path)
        node_list = pd.read_feather(node_list_path)[0].values
    except FileNotFoundError:
        return None
        # happens for SDS and DYE for some years

    # Preprocess node and link data
    nodes, pad, pad2, n = Nodes(region, year, 480, 60, 0, ratio, position, save_path)

    node_data = {
        "label": pd.DataFrame(nodes, index=node_list)["label kt"].values,
        "pad": pad2 * (n + 1) / (n - 1),
        "thickness": 2,
        "color": "#00005A",
        "x": nodes["x"].values,
        "y": nodes["y"].values,
    }

    node_data_cap = {
        "label": pd.DataFrame(nodes, index=node_list)["label t/cap"].values,
        "pad": pad2 * (n + 1) / (n - 1),
        "thickness": 2,
        "color": "#00005A",
        "x": nodes["x"].values,
        "y": nodes["y"].values,
    }

    link_data = {
        "source": data_sankey["source"],
        "target": data_sankey["target"],
        "value": data_sankey["value"],
        "label": [f"{x:.1f} kt" for x in data_sankey["value"].astype(float)],
        "color": data_sankey["color"],
    }

    # Create the Sankey diagram object
    sankey = go.Sankey(
        link=link_data,
        node=node_data,
        valueformat=".0f",
        valuesuffix=" kt",
        hoverinfo="none",
    )

    sankey_cap = go.Sankey(
        link=link_data,
        node=node_data_cap,
        valueformat=".0f",
        valuesuffix=" kt",
        hoverinfo="none",
    )

    # Precompute the figure layout
    if sankey_type in ["Commodity ownership", "Commodity all ownership"]:
        title_text = (
            sankey_subtype + " footprint of " + dictreg[region] + " in " + str(year)
        )
    elif sankey_type == "Commodity ownership pba":
        title_text = (
            sankey_subtype
            + " production and consumption of "
            + dictreg[region]
            + " in "
            + str(year)
        )
    elif sankey_type == "Commodity":
        title_text = (
            sankey_subtype + " footprint of " + dictreg[region] + " in " + str(year)
        )
    elif sankey_type == "All commodities":
        title_text = "Metal ores footprint of " + dictreg[region] + " in " + str(year)
    else:
        title_text = save_path + " " + region + " " + str(year)

    layout = dict(
        title=title_text,
        font=dict(size=10, color="black"),
        paper_bgcolor="white",
        title_x=0.5,
        title_y=0.98,
        font_family="Arial",
        autosize=False,
        width=1100,
        height=480,
        margin=dict(l=10, r=10, t=60, b=0),
    )

    # Preprocess the arrows and labels
    arrows_and_labels = preprocess_arrows_and_labels(sankey_type)

    # Save all preprocessed data together in a compressed pickle file
    preprocessed_data = {
        "sankey": sankey,
        "sankey_cap": sankey_cap,
        "layout": layout,
        "arrows_and_labels": arrows_and_labels,
    }

    if sankey_subtype is None:
        preprocessed_data_path = (
            "Results/Sankey_preprocessed/"
            + sankey_type
            + "/"
            + region
            + "_"
            + str(year)
            + ".pkl.lzma"
        )
    else:
        preprocessed_data_path = (
            "Results/Sankey_preprocessed/"
            + sankey_type
            + "/"
            + sankey_subtype
            + "/"
            + region
            + "_"
            + str(year)
            + ".pkl.lzma"
        )

    with lzma.open(preprocessed_data_path, "wb") as f:
        pickle.dump(preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Preprocessed data saved to {preprocessed_data_path}")


color_mapping = {
    "#0072ff": "#4C72B0",  # Deep Blue
    "#00cafe": "#55A868",  # Green
    "#b0ebff": "#C44E52",  # Red
    "#fff1b7": "#8172B3",  # Purple
    "#ffdc23": "#CCB974",  # Mustard Yellow
    "#ffb758": "#64B5CD",  # Cyan
    "#ff8200": "#8C8C8C",  # Gray
    "#0072ff": "#E377C2",  # Pink
    "#00cafe": "#F39C12",  # Orange
    "#b0ebff": "#17BECF",  # Sky Blue
}


def fig_sankey(
    region, year, unit="kt", sankey_type="All commodities", sankey_subtype=None
):
    """Builds a Sankey diagram for a given region and year."""

    # Construct the path to the preprocessed data
    base_path = "Results/Sankey_preprocessed/"
    subpath = f"{sankey_type}/" + (f"{sankey_subtype}/" if sankey_subtype else "")
    preprocessed_data_path = f"{base_path}{subpath}{region}_{year}.pkl.lzma"
    preprocessed_data_path = DATA_PATH.joinpath(preprocessed_data_path)

    # Load preprocessed data
    with lzma.open(preprocessed_data_path, "rb") as f:
        preprocessed_data = pickle.load(f)

    # Select Sankey data based on the unit
    sankey = (
        preprocessed_data["sankey_cap"]
        if unit == "t/cap"
        else preprocessed_data["sankey"]
    )
    layout = preprocessed_data["layout"]
    arrows_and_labels = preprocessed_data["arrows_and_labels"]

    # Update colors in the Sankey diagram
    sankey["link"]["color"] = np.array(
        [color_mapping.get(c, c) for c in sankey["link"]["color"]]
    )

    # Create the Sankey figure
    fig = go.Figure(sankey)
    fig.update_layout(**layout)
    fig.update_layout(
        shapes=arrows_and_labels["shapes"], annotations=arrows_and_labels["annotations"]
    )

    # Define legends based on the Sankey type
    if sankey_type in ["Commodity", "Commodity all ownership", "All commodities"]:
        legend_data = {
            "Commodity": {
                "colors": [
                    "white",
                    "#4C72B0",
                    "#55A868",
                    "#C44E52",
                    "#8172B3",
                    "#CCB974",
                    "#64B5CD",
                    "#8C8C8C",
                    "#E377C2",
                ],
                "names": [
                    "<b>Region of ores extraction:</b>",
                    dictreg[region],
                    "Africa",
                    "Asia-Pacific",
                    "EECCA",
                    "Europe",
                    "Latin America",
                    "Middle East",
                    "North America",
                ],
            },
            "Commodity all ownership": {
                "colors": [
                    "white",
                    "#4C72B0",
                    "#55A868",
                    "#C44E52",
                    "#8172B3",
                    "#CCB974",
                    "#64B5CD",
                    "#8C8C8C",
                    "#E377C2",
                    "#F39C12",
                ],
                "names": [
                    "<b>Nationality of mine owners:</b>",
                    dictreg[region],
                    "Africa",
                    "Asia-Pacific",
                    "EECCA",
                    "Europe",
                    "Latin America",
                    "Middle East",
                    "North America",
                    "Unknown",
                ],
            },
            "All commodities": {
                "colors": [
                    "white",
                    "#4C72B0",
                    "#55A868",
                    "#C44E52",
                    "#8172B3",
                    "#CCB974",
                    "#64B5CD",
                    "#8C8C8C",
                    "#E377C2",
                ],
                "names": [
                    "<b>Metal ores:</b>",
                    "Aluminium ores",
                    "Chromium ores",
                    "Copper ores",
                    "Gold ores",
                    "Iron ores",
                    "Lead ores",
                    "Manganese ores",
                    "Nickel ores",
                ],
                "colors2": [
                    "white",
                    "#F39C12",
                    "#17BECF",
                    "#9E9E9E",
                    "#F1A340",
                    "#D84A6B",
                    "#5E4FA2",
                    "#2C7BB6",
                ],
                "names2": [
                    "",
                    "Other metal ores",
                    "Platinum ores",
                    "Silver ores",
                    "Tin ores",
                    "Titanium ores",
                    "Uranium ores",
                    "Zinc ores",
                ],
            },
        }

        # Generate Scatter traces for the legend
        def create_legend(colors, names):
            return [
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=clr),
                    showlegend=True,
                    name=nm,
                )
                for clr, nm in zip(colors, names)
            ]

        legend_info = legend_data[sankey_type]
        legend_traces = create_legend(legend_info["colors"], legend_info["names"])
        fig.add_traces(legend_traces)

        yleg = -0.06
        if sankey_type == "All commodities":
            legend_traces2 = create_legend(
                legend_info["colors2"], legend_info["names2"]
            )
            fig.add_traces(legend_traces2)

            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5
                ),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            )
            yleg = -0.12

        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=yleg, xanchor="center", x=0.5
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )

    return fig


def save_sankey(sankey_type, sankey_subtype):
    save_path = "Results/Sankey figs/"
    ensure_directory_exists(save_path)
    sankey_type = "All commodities"
    ensure_directory_exists(save_path + sankey_type)
    sankey_subtype = None
    for region in region_acronyms:
        for year in range(2000, 2023, 1):
            fig = fig_sankey(year, region, sankey_type, sankey_subtype)
            ensure_directory_exists(save_path + sankey_type + "/" + dictreg[region])
            fig.write_image(
                save_path
                + sankey_type
                + "/"
                + dictreg[region]
                + "/"
                + sankey_type
                + "_"
                + sankey_subtype
                + "_"
                + region
                + "_"
                + str(year)
                + ".svg",
                engine="orca",
            )
