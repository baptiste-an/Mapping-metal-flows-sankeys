import pandas as pd
import numpy as np
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import lzma
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gloria_sankeys import *
from openpyxl import load_workbook
import plotly.io as pio

"""This code has been used to generate the figures for the paper together with the supplementary data file."""

path = "Results/For paper/supplementary_data.xlsx"
book = load_workbook(path)


def dicts_and_lists():
    region_dict_inv = dict(zip(regions["Region_acronyms"], regions["Region_names"]))
    metal_list = [
        "Iron",
        "Uranium",
        "Aluminium",
        "Copper",
        "Gold",
        "Lead",
        "Zinc",
        "Silver",
        "Nickel",
        "Tin",
    ]

    dict_metal_to_sect2 = {
        "Iron": "Basic iron and steel",
        "Uranium": "Basic uranium",
        "Aluminium": "Basic aluminium",
        "Copper": "Basic Copper",
        "Gold": "Basic Gold",
        "Lead": "Basic lead/zinc/silver",
        "Zinc": "Basic lead/zinc/silver",
        "Silver": "Basic lead/zinc/silver",
        "Nickel": "Basic nickel",
        "Tin": "Basic tin",
    }
    dict_ext_rm_to_metal = {
        "Iron ores concentrates and compounds": "Iron",
        "Uranium ores concentrates and compounds": "Uranium",
        "Bauxite and other aluminium ores concentrates and compounds": "Aluminium",
        "Copper ores concentrates and compounds": "Copper",
        "Gold ores concentrates and compounds": "Gold",
        "Lead ores concentrates and compounds": "Lead",
        "Zinc ores concentrates and compounds": "Zinc",
        "Silver ores concentrates and compounds": "Silver",
        "Nickel ores concentrates and compounds": "Nickel",
        "Tin ores concentrates and compounds": "Tin",
        "Manganese ores concentrates and compounds": "Manganese",
        "Iron ores": "Iron",
        "Uranium ores": "Uranium",
        "Bauxite and other aluminium ores": "Aluminium",
        "Copper ores": "Copper",
        "Gold ores": "Gold",
        "Lead ores": "Lead",
        "Zinc ores": "Zinc",
        "Silver ores": "Silver",
        "Nickel ores": "Nickel",
        "Tin ores": "Tin",
    }
    ext_RM = [
        "Material",
        [
            "Iron ores concentrates and compounds",
            "Silver ores concentrates and compounds",
            "Bauxite and other aluminium ores concentrates and compounds",
            "Gold ores concentrates and compounds",
            "Chromium ores concentrates and compounds",
            "Copper ores concentrates and compounds",
            "Manganese ores concentrates and compounds",
            "Other metal ores concentrates and compounds nec. including mixed",
            "Nickel ores concentrates and compounds",
            "Lead ores concentrates and compounds",
            "Platinum group metal ores concentrates and compounds",
            "Tin ores concentrates and compounds",
            "Titanium ores concentrates and compounds",
            "Uranium ores concentrates and compounds",
            "Zinc ores concentrates and compounds",
        ],
    ]
    return (
        region_dict_inv,
        metal_list,
        dict_metal_to_sect2,
        dict_ext_rm_to_metal,
        ext_RM,
    )


region_dict_inv, metal_list, dict_metal_to_sect2, dict_ext_rm_to_metal, ext_RM = (
    dicts_and_lists()
)


def coverage():
    """Compares S&P data with BGS to calculate coverage rates."""
    df = pd.DataFrame()
    for commodity in [
        "Copper ores",
        "Nickel ores",
        "Lead ores",
        "Zinc ores",
        "Tin ores",
        "Uranium ores",
        "Gold ores",
        "Silver ores",
        "Iron ores",
        "Bauxite and other aluminium ores",
    ]:
        own2 = feather.read_feather(
            "DATA/ownership_processed/prepro_" + commodity + ".feather"
        )
        bgs = feather.read_feather(
            "DATA/ownership_processed/bgsprepro_" + commodity + ".feather"
        )
        if commodity in ["Silver ores", "Gold ores"]:
            df[commodity] = own2.stack().sum() / bgs.sum()
        else:
            df[commodity] = own2.stack().sum() / bgs.sum()
    # to excel
    df.to_excel("Results/For paper/table_SI_coverage.xlsx")


def exports(year):
    """Calculates flows from A to B for figure 5."""

    zip_file_path = (
        gloria_path + "GLORIA_MRIOs_" + str(version) + "_" + str(year) + ".zip"
    )
    unzipped_file_path = zip_file_path[:-4]
    date = date_dict(year, version, "T1")[0]
    T_name = f"{date}_120secMother_AllCountries_002_T-Results_{year}_0{version}_Markup001(full).parquet"
    T = pd.read_parquet(unzipped_file_path + "/" + T_name)
    T_shares = T.div(T.sum(axis=1), axis=0)

    unzipped_file_path_sat = (
        gloria_path
        + "GLORIA_SatelliteAccounts_0"
        + str(version)
        + "_"
        + str(year)
        + ".zip"
    )[:-4]
    date_sat = date_dict(year, version)[1]
    TQ_name = f"{date_sat}_120secMother_AllCountries_002_TQ-Results_{year}_0{version}_Markup001(full).parquet"
    TQ = (
        pd.read_parquet(unzipped_file_path_sat + "/" + TQ_name)
        .loc[ext_RM[0]]
        .loc[ext_RM[1]]
    )

    ores_exp_year = pd.DataFrame()
    metals_exp_year = pd.DataFrame()
    for metal in metal_list:
        if metal == "Aluminium":
            ores_A_B = T_shares.xs("Aluminium ore", level=1).mul(
                TQ.loc[
                    "Bauxite and other aluminium ores concentrates and compounds"
                ].xs("Aluminium ore", level=1),
                axis=0,
            )
        elif metal in ["Lead", "Zinc", "Silver"]:
            ores_A_B = T_shares.xs("Lead/zinc/silver ores", level=1).mul(
                TQ.loc[metal + " ores concentrates and compounds"].xs(
                    "Lead/zinc/silver ores", level=1
                ),
                axis=0,
            )
        else:
            ores_A_B = T_shares.xs(metal + " ores", level=1).mul(
                TQ.loc[metal + " ores concentrates and compounds"].xs(
                    metal + " ores", level=1
                ),
                axis=0,
            )
        ores_A_B = ores_A_B.groupby(level=0, axis=1).sum()
        ores_exp_year[metal] = ores_A_B.stack()

        if metal != "Uranium":
            ores_A_staying_in_A = pd.DataFrame(
                [ores_A_B.loc[i, i] for i in ores_A_B.index], index=ores_A_B.index
            )[0]

            metal_exp = (
                T.xs(dict_metal_to_sect2[metal], level=1).groupby(level=0, axis=1).sum()
            )
            share_metal_exp = metal_exp.div(metal_exp.sum(axis=1), axis=0)
            metals_A_B = share_metal_exp * ores_A_staying_in_A
            metals_exp_year[metal] = metals_A_B.stack()
    ores_exp_year.index.names = ["region prod", "region cons"]
    metals_exp_year.index.names = ["region prod", "region cons"]

    SLY = pd.read_parquet(
        gloria_path + "SLY_" + str(version) + "/sect_agg" + str(year) + ".parquet"
    )[0]
    SLY = SLY.unstack(level=["LY name", "region cons", "sector cons"])
    SLY_agg = SLY.groupby(level=1, axis=1).sum().groupby(level=[0, 1]).sum()
    ores_L_exp_year = (
        SLY_agg.stack()
        .unstack(level=0)
        .rename(columns=dict_ext_rm_to_metal)[metal_list]
    )

    return ores_exp_year, metals_exp_year, ores_L_exp_year


def exports_all():
    """Calculates flows from A to B for figure 5 for all years."""
    ores_exp = pd.DataFrame()
    metals_exp = pd.DataFrame()
    ores_L_exp = pd.DataFrame()
    for year in range(2000, 2023, 1):
        ores_exp_year, metals_exp_year, ores_L_exp_year = exports(year)
        ores_exp[year] = ores_exp_year.stack()
        metals_exp[year] = metals_exp_year.stack()
        ores_L_exp[year] = ores_L_exp_year.stack()
    # to feather
    ores_exp.to_feather("Results/For paper/ores_exp.feather")
    metals_exp.to_feather("Results/For paper/metals_exp.feather")
    ores_L_exp.to_feather("Results/For paper/ores_L_exp.feather")


def pba_cba():
    """Calculates vectors of production in production based account and consumption based account."""
    vect_pba = pd.DataFrame()
    vect_cba = pd.DataFrame()
    for year in range(2000, 2023, 1):
        SLY = pd.read_parquet(
            gloria_path + "SLY_" + str(version) + "/sect_agg" + str(year) + ".parquet"
        )[0]
        SLY = SLY.unstack(level=["LY name", "region cons", "sector cons"])
        SLY_agg = SLY.groupby(level=1, axis=1).sum().groupby(level=[0, 1]).sum()
        vect_pba[year] = pd.DataFrame(
            SLY_agg.sum(axis=1)
            .unstack(level=0)
            .rename(columns=dict_ext_rm_to_metal)[metal_list],
            index=region_acronyms,
        ).stack(dropna=False)
        vect_cba[year] = pd.DataFrame(
            SLY_agg.groupby(level=0)
            .sum()
            .T.rename(columns=dict_ext_rm_to_metal)[metal_list],
            index=region_acronyms,
        ).stack(dropna=False)
    vect_pba.columns.names = ["year"]
    vect_cba.columns.names = ["year"]
    vect_pba.to_feather("Results/For paper/vect_pba.feather")
    vect_cba.to_feather("Results/For paper/vect_cba.feather")


def ownership_all():
    """Calculates vector of ownership."""

    com_list = [
        "Copper ores",
        "Nickel ores",
        "Lead ores",
        "Zinc ores",
        "Tin ores",
        "Uranium ores",
        "Gold ores",
        "Silver ores",
        "Iron ores",
        "Bauxite and other aluminium ores",
    ]
    com_list2 = [
        "Copper",
        "Nickel",
        "Lead",
        "Zinc",
        "Tin",
        #  "Manganese ores", not a GLORIA ores sector, it is agregated in "other non-ferrous ores"
        "Uranium",
        "Gold",
        "Silver",
        "Iron",
        "Aluminium",
    ]
    ownership_share = pd.DataFrame()
    for commodity in com_list:

        own2 = feather.read_feather(
            "DATA/ownership_processed/prepro_" + commodity + ".feather"
        )
        bgs = feather.read_feather(
            "DATA/ownership_processed/bgsprepro_" + commodity + ".feather"
        )
        bgs = bgs.unstack()

        own2.columns.names = ["year", "region mine"]
        bgs.index.names = ["year", "region mine"]

        own2_share = own2.div(bgs, axis=1)  # we now have percents of bgs
        own2_share.loc["Unknown"] = (
            (1 - own2_share.sum()).abs() + (1 - own2_share.sum())
        ) / 2  # min between 0 and 1-x, in case sp>bgs
        own2_share = own2_share.div(
            own2_share.sum(), axis=1
        )  # we make sure that the sum is 1 for the cases when sp>bgs

        own2_share = pd.DataFrame(
            own2_share.unstack(),
            index=ownership_share.index.union(own2_share.unstack().index),
        )[0]
        ownership_share = pd.DataFrame(
            ownership_share,
            index=ownership_share.index.union(own2_share.index),
            columns=com_list2,
        )

        if commodity == "Bauxite and other aluminium ores":
            ownership_share["Aluminium"] = own2_share
        else:
            ownership_share[commodity[:-5]] = own2_share

    ownership_share = ownership_share.stack(dropna=False).unstack(level=2)
    ownership_share.index.names = ["year", "region prod", "satellite"]

    vect_pba = feather.read_feather("Results/For paper/vect_pba.feather")
    ownership_share = pd.DataFrame(
        ownership_share,
        index=vect_pba.stack(dropna=False).reorder_levels([2, 0, 1]).index,
    )  # only SCG is in ownership and not pba, but all unknown data

    ownership = ownership_share.mul(
        vect_pba.stack(dropna=False).reorder_levels([2, 0, 1]), axis=0
    )

    # to feather
    ownership.stack().unstack(level=0).reorder_levels([0, 2, 1]).to_feather(
        "Results/For paper/ownership_exp.feather"
    )


def ownership_fin():
    """Calculates ownership of B in A for figure 5."""

    ownership_final = pd.DataFrame()
    for commodity in [
        "Copper ores",
        "Nickel ores",
        "Lead ores",
        "Zinc ores",
        "Tin ores",
        "Manganese ores",
        "Uranium ores",
        "Gold ores",
        "Silver ores",
        "Iron ores",
        "Bauxite and other aluminium ores",
    ]:

        own2 = feather.read_feather(
            "DATA/ownership_processed/prepro_" + commodity + ".feather"
        )
        bgs = feather.read_feather(
            "DATA/ownership_processed/bgsprepro_" + commodity + ".feather"
        )
        bgs = bgs.unstack()

        own2.columns.names = ["year", "region mine"]
        bgs.index.names = ["year", "region mine"]

        own2_share = own2.div(bgs, axis=1)  # we now have percents of bgs
        own2_share.loc["Unknown"] = (
            (1 - own2_share.sum()).abs() + (1 - own2_share.sum())
        ) / 2  # min between 0 and 1-x, in case sp>bgs
        own2_share = own2_share.div(
            own2_share.sum(), axis=1
        )  # we make sure that the sum is 1 for the cases when sp>bgs

        data_years = pd.DataFrame()
        for year in range(2000, 2023, 1):
            SLY = pd.read_parquet(
                gloria_path
                + "SLY_"
                + str(version)
                + "/sect_agg"
                + str(year)
                + ".parquet"
            )[0]
            SLY = SLY.unstack(level=["LY name", "region cons", "sector cons"])
            SLY_agg = SLY.groupby(level=1, axis=1).sum().groupby(level=[0, 1]).sum()

            own3_share = pd.DataFrame(own2_share[year].T, index=region_acronyms)
            own3_share.index.names = ["region mine"]
            data = SLY_agg.loc[commodity + " concentrates and compounds"].stack(
                dropna=False
            )
            own4_share = pd.DataFrame(index=data.index, columns=own3_share.columns)

            df_dict = own3_share.stack(dropna=False).reset_index()
            df_dict["prodown"] = df_dict["region mine"] + df_dict["region owner"]
            di = df_dict.set_index("prodown")[0].to_dict()

            df = own4_share.stack(dropna=False).reset_index()
            df["prodown"] = df["region prod"] + df["region owner"]
            df = (
                df.set_index("prodown")
                .rename(index=di)
                .reset_index()
                .set_index(data.index.names + ["region owner"])
            )["prodown"].unstack("region owner")
            df.loc[df[df.sum(axis=1) == 0].index, "Unknown"] = 1  # Important ++

            data = pd.DataFrame(
                df.mul(data, axis=0).stack(dropna=False), columns=[year]
            )
            data_years = pd.concat([data_years, data])

        ownership_final = pd.concat(
            [ownership_final, pd.DataFrame(data_years.stack(), columns=[commodity])]
        )

        # to feather
        ownership_final.to_feather("Results/For paper/ownership_final.feather")


ores_exp = feather.read_feather(
    "Results/For paper/ores_exp.feather"
)  # ores exports from A to B
metals_exp = feather.read_feather(
    "Results/For paper/metals_exp.feather"
)  # metals exports from A to B
ores_L_exp = feather.read_feather(
    "Results/For paper/ores_L_exp.feather"
)  # indirect exports from A to B
ownership_exp = feather.read_feather("Results/For paper/ownership_exp.feather")

vect_pba = feather.read_feather(
    "Results/For paper/vect_pba.feather"
)  # production based account
vect_cba = feather.read_feather(
    "Results/For paper/vect_cba.feather"
)  # consumption based account
vect_own = ownership_exp.groupby(level=[1, 2]).sum()  # ownership

ownership_final = feather.read_feather("Results/For paper/ownership_final.feather")
ownership_final = (
    ownership_final.rename(columns=dict_ext_rm_to_metal)
    .groupby(level=[0, 1, 2, 3])
    .sum()
)

df = ownership_final.groupby(level=[0, 1, 3]).sum().unstack()
cons_from_prod = df.loc[df.index.get_level_values(0) == df.index.get_level_values(1)]
cons_from_prod_share = cons_from_prod.groupby(level=1).sum() / df.groupby(level=1).sum()

df = ownership_final.groupby(level=[1, 2, 3]).sum().unstack()
cons_from_own = df.loc[df.index.get_level_values(0) == df.index.get_level_values(1)]
cons_from_own_share = cons_from_own.groupby(level=1).sum() / df.groupby(level=0).sum()


##### FIGS #####


def fig1():
    region = "CHN"
    year = 2022
    sankey_type = "All commodities"
    fig = fig_sankey(region, year, sankey_type)
    fig.write_image("Results/For paper/fig1.svg", engine="orca")
    fig.write_image("Results/For paper/fig1.pdf", engine="orca")


def fig2():
    region = "ZMB"
    year = 2022
    sankey_subtype = "Copper ores"

    sankey_type = "Commodity"
    preprocessed_data_path_a = (
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
    arrows_and_labels_a = preprocess_arrows_and_labels(
        sankey_type,
        1,
        1.03,
        # arrow_head_length=0.0,
        # arrow_head_width=0.0,
        # text_size=13,
    )

    sankey_type = "Commodity all ownership"
    preprocessed_data_path_b = (
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
    arrows_and_labels_b = preprocess_arrows_and_labels(
        sankey_type,
        0.465,
        0.485,
        arrow_head_length=0.0,
        arrow_head_width=0.0,
        text_size=13,
    )

    # Function to load preprocessed data
    def load_preprocessed_data(file_path):
        with lzma.open(file_path, "rb") as f:
            preprocessed_data = pickle.load(f)

        sankey = preprocessed_data["sankey"]
        layout = preprocessed_data["layout"]
        arrows_and_labels = preprocessed_data["arrows_and_labels"]

        updated_colors = [
            color_mapping.get(color, color) for color in sankey["link"]["color"]
        ]
        sankey["link"]["color"] = np.array(updated_colors)

        return sankey, layout, arrows_and_labels

    # Load data for Sankey plot A
    sankey_a, layout_a, arrows_and_labels_NA = load_preprocessed_data(
        preprocessed_data_path_a
    )

    # Load data for Sankey plot B
    sankey_b, layout_b, arrows_and_labels_NA = load_preprocessed_data(
        preprocessed_data_path_b
    )

    # Create the subplot layout with 2 rows and 1 column
    fig = make_subplots(
        rows=2,
        cols=1,
        # subplot_titles=(layout_a["title"], layout_b["title"]),
        row_heights=[0.5, 0.5],  # Adjust row heights as needed
        vertical_spacing=0.08,
        specs=[[{"type": "sankey"}], [{"type": "sankey"}]],  # Specify the subplot type
    )

    # Create Sankey plot A
    fig_a = go.Figure(sankey_a)
    fig_a.update_layout(**layout_a)
    # fig_a.update_layout(shapes=arrows_and_labels_a["shapes"])
    fig_a.update_layout(annotations=arrows_and_labels_a["annotations"])

    # Create Sankey plot B
    fig_b = go.Figure(sankey_b)
    fig_b.update_layout(**layout_b)
    # fig_b.update_layout(shapes=arrows_and_labels_b["shapes"])
    fig_b.update_layout(annotations=arrows_and_labels_b["annotations"])

    # Add Sankey plot A to the first row
    for trace in fig_a.data:
        fig.add_trace(trace, row=1, col=1)

    # Manually add the shapes from fig_a to fig (without row and col)
    for shape in arrows_and_labels_a["shapes"]:
        fig.add_shape(shape)

    # Manually add the annotations from fig_a to fig (without row and col)
    for annotation in arrows_and_labels_a["annotations"]:
        fig.add_annotation(annotation)

    # Add Sankey plot B to the second row
    for trace in fig_b.data:
        fig.add_trace(trace, row=2, col=1)

    # Manually add the shapes from fig_b to fig (without row and col)
    for shape in arrows_and_labels_b["shapes"]:
        fig.add_shape(shape)

    # Manually add the annotations from fig_b to fig (without row and col)
    for annotation in arrows_and_labels_b["annotations"]:
        fig.add_annotation(annotation)

    # Update layout
    fig.update_layout(height=800, showlegend=False)  # Adjust height as needed

    fig.add_annotation(
        text=layout_a["title"] + " – without ownership",
        x=0.5,
        y=1.058,  # Adjust this value to move the title up
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="center",
    )

    fig.add_annotation(
        text=layout_b["title"] + " – with ownership",
        x=0.5,
        y=0.51,  # Adjust this value to move the title up
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="center",
    )

    fig.add_annotation(
        text="<b>a.</b>",
        x=0,
        y=1.058,  # Adjust this value to move the title up
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="left",
    )

    fig.add_annotation(
        text="<b>b.</b>",
        x=0,
        y=0.515,  # Adjust this value to move the title up
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="left",
    )

    fig.update_layout(
        height=1000,  # Adjust height to give more space to plots
        width=1100,
        showlegend=False,
        margin=dict(
            l=10, r=10, t=60, b=0
        ),  # Set margins to 0 (left, right) and adjust top/bottom as needed
    )

    # save fig
    fig.write_image("Results/For paper/fig2.svg", engine="orca")
    fig.write_image("Results/For paper/fig2.pdf", engine="orca")

    df_transformed = (
        dep.drop("Manganese ores", level=1)
        .drop("DYE")
        .unstack()
        .swaplevel(axis=1)
        .sort_index(axis=1)
        .rename(
            columns={
                "own": "Aggregated ownership",
                "own2": "'Real' ownership",
                "pba": "Aggregated production",
                "pba2": "'Real' production",
            }
        )
    )

    # Use the ExcelWriter with mode='a' and the openpyxl engine
    with pd.ExcelWriter(
        path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df_transformed.to_excel(writer, sheet_name="fig3")


cons_from_prod_share = cons_from_prod_share.stack(level=0, dropna=False)
cons_from_own_share = cons_from_own_share.stack(level=0, dropna=False)
cons_from_prod_share.index.names = ["region", "satellite"]
cons_from_own_share.index.names = ["region", "satellite"]
vect_pba.index.names = ["region", "satellite"]
vect_cba.index.names = ["region", "satellite"]
vect_own.index.names = ["region", "satellite"]
ind = (
    vect_pba.index.union(vect_cba.index)
    .union(vect_own.index)
    .union(cons_from_prod_share.index)
    .union(cons_from_own_share.index)
)

vect_pba = vect_pba.reindex(ind).fillna(0)
vect_cba = vect_cba.reindex(ind).fillna(0)
vect_own = vect_own.reindex(ind).fillna(0)
pba_cba_share = (vect_pba / vect_cba).replace(np.inf, 0)
own_cba_share = (vect_own / vect_cba).replace(np.inf, 0)
dep_pba = (1 - pba_cba_share).clip(lower=0) * 100  # pba aggregated dependency
dep_own = (1 - own_cba_share).clip(lower=0) * 100  # ownership aggregated dependency

cons_from_prod_share = cons_from_prod_share.reindex(ind).fillna(0)
cons_from_own_share = cons_from_own_share.reindex(ind).fillna(0)
dep_pba2 = (1 - cons_from_prod_share).clip(lower=0) * 100  # 'real' pba dependency
dep_own2 = (1 - cons_from_own_share).clip(lower=0) * 100  # 'real' ownership dependency

dep = (
    pd.concat(
        [dep_pba.stack(), dep_own.stack(), dep_pba2.stack(), dep_own2.stack()],
        axis=1,
        keys=["pba", "own", "pba2", "own2"],
    )
    .drop("Unknown")
    .xs(2022, level=2)
)
dep2 = dep.drop(
    dep[dep.sum(axis=1) == 400].index
)  # drop regions that are 100% dependent on all four indicators


def fig3():

    ncols = 5
    nrows = 2

    # Initialize subplot parameters
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 4))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each commodity
    for i, commodity in enumerate(
        [
            "Aluminium",
            "Copper",
            "Gold",
            "Iron",
            "Lead",
            "Nickel",
            "Silver",
            "Tin",
            "Uranium",
            "Zinc",
        ]
    ):
        print(commodity)
        pba22 = vect_pba[2022].replace(0, np.nan).dropna()
        own22 = (
            ownership_exp.loc[
                ownership_exp.index.get_level_values(level=0)
                == ownership_exp.index.get_level_values(level=1)
            ]
            .groupby(level=[1, 2])
            .sum()[2022]
        )
        own22 = pd.DataFrame(
            own22.replace(0, np.nan).dropna(), index=pba22.index
        ).replace(np.nan, 0)[2022]
        ratio = (own22 / pba22).swaplevel().sort_index().drop("SDS", level=1)

        y = ratio.loc[commodity]
        x = (gdp / pop)[2022].loc[y.index] / 1000

        ax = axes[i]
        ax.scatter(x, y, s=8)
        ax.set_title(f"{commodity}")
        ax.set_ylim(top=1.05)

        # Set x-axis label only for the bottom row
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("GDP per Capita (kUS$)")

        # Set y-axis label only for the first column
        if i % ncols == 0:
            ax.set_ylabel("Ownership Ratio")
        else:
            ax.set_ylabel("")

    # Hide any unused subplots
    for j in range(len(ratio.unstack().index), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("Results/For paper/fig3.pdf", bbox_inches="tight")
    plt.savefig("Results/For paper/fig3.svg", bbox_inches="tight")

    plt.show()

    ### save

    # y=pd.DataFrame(index=region_acronyms)
    # for i, commodity in enumerate(
    #     [
    #         "Aluminium",
    #         "Copper",
    #         "Gold",
    #         "Iron",
    #         "Lead",
    #         "Nickel",
    #         "Silver",
    #         "Tin",
    #         "Uranium",
    #         "Zinc",
    #     ]
    # ):
    #     print(commodity)
    #     pba22 = vect_pba[2022].replace(0, np.nan).dropna()
    #     own22 = (
    #         ownership_exp.loc[
    #             ownership_exp.index.get_level_values(level=0)
    #             == ownership_exp.index.get_level_values(level=1)
    #         ]
    #         .groupby(level=[1, 2])
    #         .sum()[2022]
    #     )
    #     own22 = pd.DataFrame(
    #         own22.replace(0, np.nan).dropna(), index=pba22.index
    #     ).replace(np.nan, 0)[2022]
    #     ratio = (own22 / pba22).swaplevel().sort_index().drop("SDS", level=1)

    #     y[commodity] = ratio.loc[commodity]
    # with pd.ExcelWriter(
    #     path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    # ) as writer:
    #     y.dropna(how='all',axis=0).to_excel(writer, sheet_name="fig3")


def fig4():
    df = dep2

    # Set figure dimensions and font sizes
    w = 6.4  # Double the width of a single subplot for side-by-side plots
    h = 8
    fs1 = 8  # Small font size for tick labels
    fs2 = 8  # Font size for titles and axis labels
    fs3 = 10
    fstt = 9

    # Create a single figure with 5x2 subplots for each side-by-side plot
    fig, axs = plt.subplots(5, 4, figsize=(w, h))

    # Plot (a) on the left (first 2 columns)
    for i, commodity in enumerate(metal_list):
        x = df.xs(commodity, level=1)["pba"]
        y = df.xs(commodity, level=1)["own"]

        sns.kdeplot(
            x=x,
            y=y,
            ax=axs[i % 5, i // 5],
            fill=True,
            cmap="Blues",
            thresh=0,
        )

        axs[i % 5, i // 5].set_title(commodity, fontsize=fs3)
        axs[i % 5, i // 5].set_ylim(-2, 102)
        axs[i % 5, i // 5].set_xlim(-2, 102)
        axs[i % 5, i // 5].tick_params(axis="both", which="major", labelsize=fs1)

        if (i // 5) in [0]:
            axs[i % 5, i // 5].set_ylabel("Ownership dependency", fontsize=fs2)
        else:
            axs[i % 5, i // 5].set_ylabel("", fontsize=fs2)
        if i % 5 == 4:
            axs[i % 5, i // 5].set_xlabel("Production dependency", fontsize=fs2)
        else:
            axs[i % 5, i // 5].set_xlabel("", fontsize=fs2)

    # Plot (b) on the right (last 2 columns)
    for i, commodity in enumerate(metal_list):
        x = df.xs(commodity, level=1)["pba2"]
        y = df.xs(commodity, level=1)["own2"]

        sns.kdeplot(
            x=x,
            y=y,
            ax=axs[i % 5, 2 + (i // 5)],  # Shift to right side
            fill=True,
            cmap="Blues",
            thresh=0,
        )

        axs[i % 5, 2 + (i // 5)].set_title(commodity, fontsize=fs3)
        axs[i % 5, 2 + (i // 5)].set_ylim(-2, 102)
        axs[i % 5, 2 + (i // 5)].set_xlim(-2, 102)
        axs[i % 5, 2 + (i // 5)].tick_params(axis="both", which="major", labelsize=fs1)

        if 2 + (i // 5) in [0]:
            axs[i % 5, 2 + (i // 5)].set_ylabel("Ownership dependency", fontsize=fs2)
        else:
            axs[i % 5, 2 + (i // 5)].set_ylabel("", fontsize=fs2)
        if i % 5 == 4:
            axs[i % 5, 2 + (i // 5)].set_xlabel("Production dependency", fontsize=fs2)
        else:
            axs[i % 5, 2 + (i // 5)].set_xlabel("", fontsize=fs2)

    plt.tight_layout()

    # Add titles and adjust layout
    # fig.suptitle('Comparison of Dependencies', fontsize=fstt, x=0.3, y=1.05)

    fig.text(0, 1, "a.", ha="left", fontsize=fstt, fontweight="bold")
    fig.text(0.25, 1, "Aggregated dependencies", ha="center", fontsize=fstt)
    fig.text(0.5, 1, "b.", ha="left", fontsize=fstt, fontweight="bold")
    fig.text(0.75, 1, "Real dependencies", ha="center", fontsize=fstt)
    # fig.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)

    # Save the combined figure
    fig.savefig("Results/For paper/fig4.pdf", bbox_inches="tight")
    fig.savefig("Results/For paper/fig4.svg", bbox_inches="tight")

    plt.show()

    df_transformed = (
        dep.drop("Manganese ores", level=1)
        .drop("DYE")
        .unstack()
        .swaplevel(axis=1)
        .sort_index(axis=1)
        .rename(
            columns={
                "own": "Aggregated ownership",
                "own2": "'Real' ownership",
                "pba": "Aggregated production",
                "pba2": "'Real' production",
            }
        )
    )

    # Use the ExcelWriter with mode='a' and the openpyxl engine
    with pd.ExcelWriter(
        path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df_transformed.to_excel(writer, sheet_name="fig4")


def fig4bis():  # this is a version without eliminating the regions that are 100% dependent on all four indicators

    df = dep

    # Set figure dimensions and font sizes
    w = 7.20472  # Double the width of a single subplot for side-by-side plots
    h = 9
    fs1 = 8  # Small font size for tick labels
    fs2 = 8  # Font size for titles and axis labels
    fs3 = 10
    fstt = 9

    # Create a single figure with 5x2 subplots for each side-by-side plot
    fig, axs = plt.subplots(5, 4, figsize=(w, h))

    # Plot (a) on the left (first 2 columns)
    for i, commodity in enumerate(metal_list):
        x = df.xs(commodity, level=1)["pba"]
        y = df.xs(commodity, level=1)["own"]

        sns.kdeplot(
            x=x,
            y=y,
            ax=axs[i % 5, i // 5],
            fill=True,
            cmap="Blues",
            thresh=0,
        )

        axs[i % 5, i // 5].set_title(commodity, fontsize=fs3)
        axs[i % 5, i // 5].set_xlabel("Production dependency", fontsize=fs2)
        axs[i % 5, i // 5].set_ylabel("Ownership dependency", fontsize=fs2)
        axs[i % 5, i // 5].set_ylim(-2, 102)
        axs[i % 5, i // 5].set_xlim(-2, 102)
        axs[i % 5, i // 5].tick_params(axis="both", which="major", labelsize=fs1)

    # Plot (b) on the right (last 2 columns)
    for i, commodity in enumerate(metal_list):
        x = df.xs(commodity, level=1)["pba2"]
        y = df.xs(commodity, level=1)["own2"]

        sns.kdeplot(
            x=x,
            y=y,
            ax=axs[i % 5, 2 + (i // 5)],  # Shift to right side
            fill=True,
            cmap="Blues",
            thresh=0,
        )

        axs[i % 5, 2 + (i // 5)].set_title(commodity, fontsize=fs3)
        axs[i % 5, 2 + (i // 5)].set_xlabel("Production dependency", fontsize=fs2)
        axs[i % 5, 2 + (i // 5)].set_ylabel("Ownership dependency", fontsize=fs2)
        axs[i % 5, 2 + (i // 5)].set_ylim(-2, 102)
        axs[i % 5, 2 + (i // 5)].set_xlim(-2, 102)
        axs[i % 5, 2 + (i // 5)].tick_params(axis="both", which="major", labelsize=fs1)

    plt.tight_layout()

    # Add titles and adjust layout
    # fig.suptitle('Comparison of Dependencies', fontsize=fstt, x=0.3, y=1.05)

    fig.text(0, 1, "a.", ha="left", fontsize=fstt, fontweight="bold")
    fig.text(0.25, 1, "Aggregated dependencies", ha="center", fontsize=fstt)
    fig.text(0.5, 1, "b.", ha="left", fontsize=fstt, fontweight="bold")
    fig.text(0.75, 1, "Real dependencies", ha="center", fontsize=fstt)
    # fig.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)

    # Save the combined figure
    fig.savefig("For paper/fig4bis.pdf", bbox_inches="tight")
    fig.savefig("For paper/fig4bis.svg", bbox_inches="tight")

    plt.show()


def fig5():

    fig, axs = plt.subplots(10, 3, figsize=(6.5, 14))

    for i, commodity in enumerate(metal_list):
        # Define datasets with special handling for Uranium
        if commodity == "Uranium":
            datasets = [
                (ownership_exp, ores_exp, "Ores", 0),
                None,  # Skip the second column for Uranium
                (ownership_exp, ores_L_exp, "Ores_L", 2),
            ]
        else:
            datasets = [
                (ownership_exp, ores_exp, "Ores", 0),
                (ownership_exp, metals_exp, "Metals", 1),
                (ownership_exp, ores_L_exp, "Ores_L", 2),
            ]

        for df_x, df_y, label, col_idx in filter(None, datasets):
            df_y = df_y.drop(
                df_y[
                    df_y.index.get_level_values(0) == df_y.index.get_level_values(1)
                ].index
            )
            x = df_x.xs(commodity, level=2).replace(0, np.nan).stack()
            y = df_y.xs(commodity, level=2).replace(0, np.nan).stack()
            log_x = np.log(x)
            log_y = np.log(y)

            # Handle infinite and NaN values
            valid_idx = (
                log_x.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .index.intersection(
                    log_y.replace([np.inf, -np.inf], np.nan).dropna().index
                )
            )
            log_x = log_x.loc[valid_idx]
            log_y = log_y.loc[valid_idx]

            if len(log_x) > 1 and len(log_y) > 1:  # Ensure enough valid data points
                r_value = np.corrcoef(log_x, log_y)[0, 1]
                r_squared = r_value**2
            else:
                r_squared = np.nan  # Not enough data to calculate a correlation

            # Plot the density plot
            sns.kdeplot(x=log_x, y=log_y, ax=axs[i, col_idx], fill=True, cmap="Blues")
            if col_idx == 0:
                axs[i, col_idx].set_title(commodity + " - ore exports", fontsize=10)
            elif col_idx == 1:
                axs[i, col_idx].set_title(commodity + " - metal exports", fontsize=10)
            else:
                axs[i, col_idx].set_title(
                    commodity + " - indirect ore exports", fontsize=10
                )

            # Set labels conditionally
            if col_idx == 0:
                axs[i, col_idx].set_ylabel("Flow from A to B", fontsize=9)
            else:
                axs[i, col_idx].set_ylabel("", fontsize=9)
            if i == 9:  # Bottom row
                axs[i, col_idx].set_xlabel("Mine control by B in A", fontsize=9)
            else:
                axs[i, col_idx].set_xlabel("", fontsize=9)

            # Add R^2 annotation to the plot
            axs[i, col_idx].annotate(
                f"$R^2 = {r_squared:.2f}$",
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                fontsize=10,
                color="black",
                ha="left",
                va="bottom",
            )
            axs[4, 1].set_yticks([0, 20])

    plt.tight_layout()
    plt.savefig("Results/For paper/fig5.pdf", bbox_inches="tight")
    plt.savefig("Results/For paper/fig5.svg", bbox_inches="tight")
    plt.show()

    # with pd.ExcelWriter(
    #     path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    # ) as writer:
    #     ownership_exp.reorder_levels([2, 0, 1]).sort_index().to_excel(
    #         writer, sheet_name="fig5_ownership_B_in_A"
    #     )
    #     ores_exp.reorder_levels([2, 0, 1]).sort_index().to_excel(
    #         writer, sheet_name="fig5_flow_ores_A_to_B"
    #     )
    #     metals_exp.reorder_levels([2, 0, 1]).sort_index().to_excel(
    #         writer, sheet_name="fig5_flow_metals_A_to_B"
    #     )
    #     ores_L_exp.reorder_levels([2, 0, 1]).sort_index().to_excel(
    #         writer, sheet_name="fig5_flow_indirect_A_to_B"
    #     )


##### ITEM 6 #####
def table1():

    df = pd.concat(
        [vect_pba[2022], vect_cba[2022], vect_own[2022]],
        axis=1,
        keys=["pba", "cba", "own"],
    ).unstack()
    df = df.div(df.sum(), axis=1).stack().swaplevel().sort_index()
    df = df[df["own"] > 0.1].rename(index=region_dict_inv)
    df.index.names = ["Commodity", "Region"]
    df.columns = ["Production", "Consumption", "Ownership"]
    df.to_excel("Results/For paper/table1.xlsx")


################# VERIFICATIONS #################


ownership_final_pba = (
    ownership_final.stack().unstack(level=3).groupby(level=[0, 3]).sum()
)
ownership_final_pba.sum().sum() / vect_pba.sum().sum()

ownership_final_cba = (
    ownership_final.stack().unstack(level=3).groupby(level=[1, 3]).sum()
)
ownership_final_cba.sum().sum() / vect_cba.sum().sum()


##### numerical data for paper #####

for i in dep2.unstack(level=0).drop("Manganese ores").index:
    print(i + " " + str(len(dep2.xs(i, level=1))))


shares = pd.DataFrame(
    index=dep2.columns, columns=dep2.unstack(level=0).drop("Manganese ores").index
)
for com in dep2.unstack(level=0).drop("Manganese ores").index:
    for cat in dep2.columns:
        df = dep2.xs(com, level=1)[cat]
        shares.loc[cat, com] = len(df[df < 90]) / 162
1 - shares.sort_index().T.mean()
