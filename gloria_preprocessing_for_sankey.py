import pyarrow.feather as feather
from gloria_preprocessing_initial import *
from joblib import Parallel, delayed
import numpy as np

"""This scripts extracts from SLY and ownership data all the info required to build every type of sankey diagram."""


def pop_gdp_agg():
    """Reads population and GDP data, and aggregates it by region and year."""
    pop = pd.read_excel("DATA/population_WB.xlsx", index_col=[0, 1, 2, 3])[:-5].loc[
        "Population, total", "SP.POP.TOTL"
    ]
    pop.columns = [int(i[:4]) for i in pop.columns]
    country_to_region = {
        "ASM": "XAM",
        "ATG": "XAM",
        "ABW": "XAM",
        "BRB": "XAM",
        "BMU": "XAM",
        "VGB": "XAM",
        "CYM": "XAM",
        "CUW": "XAM",
        "DMA": "XAM",
        "GRL": "XAM",
        "GRD": "XAM",
        "GUM": "XAM",
        "GUY": "XAM",
        "PRI": "XAM",
        "KNA": "XAM",
        "LCA": "XAM",
        "VCT": "XAM",
        "SUR": "XAM",
        "TTO": "XAM",
        "TCA": "XAM",
        "VIR": "XAM",
        "AND": "XEU",
        "CHI": "XEU",
        "FRO": "XEU",
        "GIB": "XEU",
        "IMN": "XEU",
        "XKX": "XEU",
        "LIE": "XEU",
        "MCO": "XEU",
        "MNE": "XEU",
        "SMR": "XEU",
        "COM": "XAF",
        "SWZ": "XAF",
        "GNB": "XAF",
        "LSO": "XAF",
        "STP": "XAF",
        "SYC": "XAF",
        "SSD": "XAF",
        "CPV": "XAS",
        "FJI": "XAS",
        "PYF": "XAS",
        "KIR": "XAS",
        "MAC": "XAS",
        "MDV": "XAS",
        "MHL": "XAS",
        "MUS": "XAS",
        "FSM": "XAS",
        "NRU": "XAS",
        "NCL": "XAS",
        "MNP": "XAS",
        "PLW": "XAS",
        "WSM": "XAS",
        "SXM": "XAS",
        "SLB": "XAS",
        "TLS": "XAS",
        "TON": "XAS",
        "TUV": "XAS",
        "VUT": "XAS",
        "SSD": "SDS",
    }
    pop_agg = (
        pop.rename(country_to_region)
        .groupby(level=1)
        .sum()[[i for i in range(1990, 2023, 1)]]
    )

    gdp = pd.read_excel("DATA/gdp_WB.xlsx", index_col=[0, 1, 2, 3])[:-5].loc[
        "GDP (current US$)", "NY.GDP.MKTP.CD"
    ]
    gdp.columns = [int(i[:4]) for i in gdp.columns]
    gdp_agg = (
        gdp.rename(country_to_region)
        .replace("..", np.nan)
        .groupby(level=1)
        .sum()[[i for i in range(1990, 2023, 1)]]
    )
    # careful, a few data points are missing: gdp.rename(country_to_region).swaplevel().loc[["XAM","XEU","XAF","XAS"]][[i for i in range(2000,2023,1)]]
    return pop_agg, gdp_agg


pop, gdp = pop_gdp_agg()


def ownership_prepro():
    """Reads BGS and S&P data, and saves it into common feather format."""
    ownership_all = pd.read_excel(
        "DATA/ownership_for_sankeys.xlsx",
        index_col=[0, 1],
        header=[0, 1],
    )
    bgs_all = pd.read_excel("DATA/bgs_1970_2022.xlsx", index_col=[0, 1])

    for commodity, ore in zip(
        [
            "Copper",
            "Nickel",
            "Lead",
            "Zinc",
            "Tin",
            "Manganese",
            "Uranium",
            "Gold",
            "Silver",
            "Iron",
            "Aluminium",
        ],
        [
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
        ],
    ):

        ownership_sp = ownership_all[commodity]
        bgs = bgs_all[commodity].unstack()
        ownership_sp.index.names = ["region mine", "region owner"]
        own = ownership_sp.groupby(level=[0, 1]).sum()
        own = own[own.sum(axis=1) > 0]
        bgs = bgs[own.columns]
        bgs = bgs[bgs.sum(axis=1) > 0]
        bgs = bgs.rename(index={"NCL": "FRA", "XKX": "XEU"}).groupby(level=0).sum()

        own_agg = own.groupby(level=0).sum()
        ind = own_agg.index.intersection(bgs.index)
        own2 = own.loc[ind].unstack(level=0)

        # save
        own2.to_feather("DATA/ownership_processed/prepro_" + ore + ".feather")
        bgs.to_feather("DATA/ownership_processed/bgsprepro_" + ore + ".feather")


ownership_prepro()


def SLY_agg(year, version=59):
    """Calculates SLYagg with several combinations of S, L, and Y, and saves results."""

    # Ensure required directories exist
    base_path = f"{gloria_path}SLY_{version}/sect_agg"
    os.makedirs(base_path, exist_ok=True)

    # Load data only once
    date, date_sat = date_dict(year, version)
    conc_sec_cons = pd.read_excel(
        f"DATA/concordance_{version}.xlsx",
        sheet_name="sector cons",
        index_col=0,
    )
    conc_sec_prod = pd.read_excel(
        f"DATA/concordance_{version}.xlsx",
        sheet_name="sector prod",
        index_col=0,
    )
    dict_sect_prod = conc_sec_prod["sector prod agg"].to_dict()
    dict_sect_cons = conc_sec_cons["sector cons agg"].to_dict()

    sectors = pd.read_excel(f"DATA/GLORIA_ReadMe_0{version}.xlsx", sheet_name="Sectors")
    sector_names = sectors["Sector_names"].values

    # Generate multi-index for identity matrix
    regions = region_acronyms.tolist()
    multi_index_short = pd.MultiIndex.from_product(
        [regions, sector_names], names=["Region", "Sector"]
    )

    df = pd.DataFrame(1, index=multi_index_short, columns=multi_index_short)
    mask = (
        df.index.get_level_values(1).values[:, None]
        == df.columns.get_level_values(1).values
    )
    df.values[~mask] = 0
    id_sect = df.copy()

    Y_path = f"{gloria_path}GLORIA_MRIOs_{version}_{year}/{date}_120secMother_AllCountries_002_Y-Results_{year}_0{version}_Markup001(full).parquet"
    L_path = f"{gloria_path}GLORIA_MRIOs_{version}_{year}/{date}_120secMother_AllCountries_002_L-Results_{year}_0{version}_Markup001(full).parquet"
    S_path = f"{gloria_path}GLORIA_SatelliteAccounts_0{version}_{year}/{date_sat}_120secMother_AllCountries_002_S-Results_{year}_0{version}_Markup001(full).parquet"

    Y = pd.read_parquet(Y_path)
    L = pd.read_parquet(L_path)
    S = pd.read_parquet(S_path)

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

    S2 = S.loc[ext_RM[0]].loc[ext_RM[1]]

    SLY = pd.DataFrame()
    for fd_cat in Y.columns.get_level_values(
        level=1
    ).unique():  # 31 minutes, 15 extensions, most time in L.dot(Ycat_diag)
        Ycat = Y.xs(fd_cat, level=1, axis=1)
        Ycat_diag = (
            pd.concat(
                [Ycat] * 120,
                keys=Ycat.index.get_level_values(level=1).unique(),
                axis=1,
            )
            .swaplevel(axis=1)
            .sort_index(axis=1)
        )
        Ycat_diag = pd.DataFrame(Ycat_diag, index=Y.index, columns=Y.index)
        Ycat_diag = Ycat_diag * id_sect

        LYcat = L.dot(Ycat_diag)  # 4 min
        if len(conc_sec_cons.columns) > 1:
            LYcat_agg = (
                LYcat.stack(level=0)
                .dot(conc_sec_cons.replace(np.nan, 0))
                .unstack()
                .swaplevel(axis=1)
            )
        else:
            LYcat_agg = (
                LYcat.rename(columns=dict_sect_cons).groupby(level=[0, 1], axis=1).sum()
            )

        SLYcat_agg = pd.concat([LYcat_agg] * len(S2), keys=S2.index).mul(
            S2.swaplevel(axis=1).stack(dropna=False).stack(dropna=False), axis=0
        )

        current_index = SLYcat_agg.index

        # Create a new third level using the dictionary, handling potential duplicates
        new_level_3 = current_index.get_level_values(2).map(dict_sect_prod)

        # Combine the new third level with the other levels
        new_index = pd.MultiIndex.from_arrays(
            [
                current_index.get_level_values(0),
                current_index.get_level_values(1),
                new_level_3,
            ]
        )
        SLYcat_agg.index = new_index
        SLYcat_agg_agg = SLYcat_agg.groupby(level=[0, 1, 2]).sum()
        SLY[fd_cat] = SLYcat_agg_agg.stack(dropna=False).stack(dropna=False)

        del Ycat, Ycat_diag, LYcat, LYcat_agg, SLYcat_agg, SLYcat_agg_agg

    SLY.index.names = [
        "satellite",
        "region prod",
        "sector prod",
        "sector cons",
        "region cons",
    ]
    SLY.columns.names = ["LY name"]
    SLY = pd.DataFrame(SLY.replace(0, np.nan).stack())

    SLY.to_parquet(
        gloria_path + "SLY_" + str(version) + "/sect_agg" + str(year) + ".parquet"
    )
    del Y, L, S, SLY


def SLY_agg_reg(year, version=59):
    """Aggregates SLY to select only the data relevant to a given region."""
    conc_reg_cons = pd.read_excel(
        "DATA/concordance_" + str(version) + ".xlsx",
        sheet_name="region cons",
        index_col=0,
    )
    conc_reg_prod = pd.read_excel(
        "DATA/concordance_" + str(version) + ".xlsx",
        sheet_name="region prod",
        index_col=0,
    )
    SLY = pd.read_parquet(
        gloria_path + "SLY_" + str(version) + "/sect_agg" + str(year) + ".parquet"
    )[0]
    SLY = SLY.unstack(level=["LY name", "region cons", "sector cons"])
    for region in region_acronyms:
        ensure_directory_exists(gloria_path + "SLY_" + str(version) + "/" + region)
        dict_reg_cons = conc_reg_cons.drop(region)["region cons"].to_dict()
        dict_reg_prod = conc_reg_prod.drop(region)["region prod"].to_dict()
        SLY_reg = (
            SLY.rename(index=dict_reg_prod)
            .groupby(level=[0, 1, 2])
            .sum()
            .rename(columns=dict_reg_cons)
            .groupby(level=[0, 1, 2], axis=1)
            .sum()
        )

        data = SLY_reg.stack(dropna=False).stack(dropna=False)
        data = (
            pd.DataFrame(data.stack())
            .rename(columns={0: "value"})
            .reorder_levels([0, 1, 2, 3, 5, 4])
        )

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
            "Aluminium ores",
        ]:
            if commodity == "Aluminium ores":
                data_own = add_ownership(
                    data.loc[
                        ["Bauxite and other aluminium ores concentrates and compounds"]
                    ],
                    region,
                    year,
                    commodity,
                )
            else:
                data_own = add_ownership(
                    data.loc[[commodity + " concentrates and compounds"]],
                    region,
                    year,
                    commodity,
                )
            mask = (
                (data_own.index.get_level_values(level=1) == region)
                | (data_own.index.get_level_values(level=5) == region)
                | (data_own.index.get_level_values(level=6) == region)
            )
            data_own = data_own[mask]
            data_own.to_feather(
                gloria_path
                + "SLY_"
                + str(version)
                + "/"
                + region
                + "/"
                + commodity
                + "_ownership_"
                + str(year)
                + ".feather"
            )
        ###
        mask = (data.index.get_level_values(level=1) == region) | (
            data.index.get_level_values(level=5) == region
        )
        data = data[mask]
        data.to_feather(
            gloria_path
            + "SLY_"
            + str(version)
            + "/"
            + region
            + "/"
            + str(year)
            + ".feather"
        )


def add_ownership(data, region, year, commodity):
    """Adds ownership data to SLY data."""

    conc_reg_prod = pd.read_excel(
        "DATA/concordance_" + str(version) + ".xlsx",
        sheet_name="region prod",
        index_col=0,
    )
    dict_reg_prod = conc_reg_prod.drop(region)["region prod"].to_dict()

    if commodity == "Aluminium ores":
        own2 = feather.read_feather(
            "DATA/ownership_processed/prepro_Bauxite and other aluminium ores.feather"
        )
        bgs = feather.read_feather(
            "DATA/ownership_processed/bgsprepro_Bauxite and other aluminium ores.feather"
        )
    else:

        own2 = feather.read_feather(
            "DATA/ownership_processed/prepro_" + commodity + ".feather"
        )
        bgs = feather.read_feather(
            "DATA/ownership_processed/bgsprepro_" + commodity + ".feather"
        )

    own2_reg_agg = (
        own2.stack().rename(index=dict_reg_prod).groupby(level=[0, 1]).sum().unstack()
    )
    reg2 = conc_reg_prod["region prod"].unique().tolist() + [region]
    own2_reg_agg = (
        pd.DataFrame(own2_reg_agg.stack(level=0), columns=reg2)
        .unstack()
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
    own2_reg_agg.columns.names = ["year", "region mine"]
    bgs_reg_agg = bgs.rename(index=dict_reg_prod).groupby(level=[0]).sum()
    bgs_reg_agg = pd.DataFrame(bgs_reg_agg, index=reg2).unstack()
    bgs_reg_agg.index.names = ["year", "region mine"]

    own2_share = own2_reg_agg.div(bgs_reg_agg, axis=1)  # we now have percents of bgs
    own2_share.loc["Unknown"] = (
        (1 - own2_share.sum()).abs() + (1 - own2_share.sum())
    ) / 2  # min between 0 and 1-x, in case sp>bgs
    own2_share = own2_share.div(
        own2_share.sum(), axis=1
    )  # we make sure that the sum is 1 for the cases when sp>bgs

    own3_share = own2_share[year].T
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

    data = pd.DataFrame(
        df.mul(data["value"], axis=0).stack(dropna=False), columns=["value"]
    )

    return data


####


def constants():
    regions = pd.read_excel(
        gloria_path + "GLORIA_ReadMe_059.xlsx", sheet_name="Regions"
    )
    region_acronyms = regions["Region_acronyms"].values
    dictreg = dict(zip(regions["Region_acronyms"], regions["Region_names"]))
    dict_with_spaces = {k + " ": str(v) for k, v in dictreg.items()}
    dict_with_spaces2 = {k + "  ": str(v) for k, v in dictreg.items()}
    dict_with_spaces3 = {k + "   ": str(v) for k, v in dictreg.items()}
    dict_with_spaces4 = {k + "    ": str(v) for k, v in dictreg.items()}

    dictreg = {
        **dictreg,
        **dict_with_spaces,
        **dict_with_spaces2,
        **dict_with_spaces3,
        **dict_with_spaces4,
    }

    Dict_cba = dict(
        {
            "Acquisitions less disposals of valuables P.53": "Acquisitions less disposals of valuables",
            "Changes in inventories P.52": "Inventories",
            "Government final consumption P.3g": "Government",
            "Gross fixed capital formation P.51": "GFCF",
            "Household final consumption P.3h": "Households",
            "Non-profit institutions serving households P.3n": "NPISHS",
        }
    )

    colors = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#CCB974",
        "#64B5CD",
        "#8C8C8C",
        "#E377C2",
        "#F39C12",
        "#17BECF",
        "#9E9E9E",
        "#F1A340",
        "#D84A6B",
        "#5E4FA2",
        "#2C7BB6",
    ]

    ext_RM = [
        "Material",
        [
            "Iron ores",
            "Silver ores",
            "Bauxite and other aluminium ores - gross ore",
            "Gold ores",
            "Chromium ores",
            "Copper ores",
            "Manganese ores",
            "Other metal ores",
            "Nickel ores",
            "Lead ores",
            "Platinum group metal ores",
            "Tin ores",
            "Titanium ores",
            "Uranium ores",
            "Zinc ores",
        ],
    ]

    return region_acronyms, dictreg, Dict_cba, colors, ext_RM


region_acronyms, dictreg, Dict_cba, colors, ext_RM = constants()


def variables1(sankey_type, sankey_subtype=None):

    if sankey_type == "All commodities":
        position = [
            "0. extension",
            "1. imp reg",
            "2. cba reg",
            "3. cba fd",
            "4. cba sect",
        ]
        save_path = "Results/All_commodities"
        ensure_directory_exists(save_path)
    elif sankey_type == "Commodity":
        position = [
            "0. imp reg",
            "1. cba reg",
            "2. cba fd",
            "3. cba sect",
        ]
        save_path = "Results/Commodity"
        ensure_directory_exists(save_path)
    elif sankey_type == "Commodity all ownership":
        position = [
            "0. region owner",
            "1. imp reg",
            "2. cba reg",
            "3. cba fd",
            "4. cba sect",
        ]
        save_path = "Results/commodity all ownership/" + sankey_subtype
        ensure_directory_exists(save_path)

    return position, save_path


def variables2(region):
    """Defines some of the variables necessary to build sankey."""

    DictRoW = dict.fromkeys(
        [
            "Africa",
            "Asia-Pacific",
            "EECCA",
            "Europe",
            "Latin America",
            "North America",
            "Middle East",
        ],
        "RoW - ",
    )
    DictRoW[region] = ""
    DictRoW[dictreg[region]] = ""  # dictreg to full name

    DictExp = dict.fromkeys(
        [
            "RoW - Acquisitions less disposals of valuables",
            "RoW - Inventories",
            # "RoW - Government",
            #  "RoW - GFCF",
            #  "RoW - Households",
            "RoW - NPISHS",
        ],
        "RoW - Others",  # Exports
    )

    return DictRoW, DictExp


######


def add_steps(data, region, sankey_type):
    """Adds columns to data for each step of the sankey diagram."""
    DictRoW, DictExp = variables2(region)

    ext = data.index.get_level_values(level="satellite")
    imp_reg = data.index.get_level_values(level="region prod") + " "
    cba_fd = (
        data.rename(index=DictRoW).index.get_level_values(level="region cons")
        + data.rename(index=Dict_cba).index.get_level_values(level="LY name")
    ).map(lambda x: DictExp.get(x, x))
    cba_sect = data.rename(index=DictRoW).index.get_level_values(
        level="region cons"
    ) + data.index.get_level_values(level="sector cons")
    cba_reg = data.index.get_level_values(level="region cons") + "   "

    if sankey_type == "All commodities":
        data["0. extension"] = ext
        data["1. imp reg"] = imp_reg
        data["2. cba reg"] = cba_reg
        data["3. cba fd"] = cba_fd
        data["4. cba sect"] = cba_sect
    elif sankey_type == "Commodity":
        data["0. imp reg"] = imp_reg
        data["1. cba reg"] = cba_reg
        data["2. cba fd"] = cba_fd
        data["3. cba sect"] = cba_sect
    elif sankey_type == "Commodity all ownership":
        data["0. region owner"] = (
            data.index.get_level_values(level="region owner") + "  "
        )
        data["1. imp reg"] = imp_reg
        data["2. cba reg"] = cba_reg
        data["3. cba fd"] = cba_fd
        data["4. cba sect"] = cba_sect

    return data


def data_STV(data, region, node_dict, position, sankey_type):
    """Adds "source", "target", "value", 'color' to data_sankey to represnet the flows between nodes."""

    # STV = source, target, value

    dict_col_level = dict(
        {
            "All commodities": "0. extension",
            "Commodity": "0. imp reg",
            "Commodity all ownership": "0. region owner",
        }
    )

    data_sankey = pd.DataFrame()
    for j in range(0, len(position) - 1, 1):
        data_j = pd.DataFrame()
        data_j["source"] = data[position[j]].replace(node_dict)
        data_j["target"] = data[position[j + 1]].replace(node_dict)
        data_j["value"] = data["value"]
        data_j["color label"] = data[dict_col_level[sankey_type]]
        data_j["position"] = position[j]
        data_sankey[j] = data_j.stack(dropna=False)

    data_sankey = data_sankey.unstack().stack(level=0)
    data_sankey = data_sankey[["source", "target", "value", "color label", "position"]]

    if sankey_type in ["All commodities", "Commodity"]:
        data_sankey = (
            data_sankey.reset_index()
            .set_index(
                [
                    "sector cons",
                    "sector prod",
                    "region prod",
                    "satellite",
                    "LY name",
                    "region cons",
                ]
            )
            .drop("level_6", axis=1)
        )

    if dict_col_level[sankey_type] == "0. region owner":
        if (region + "  ") in data_sankey["color label"].unique().tolist():
            colors_lab_names = data_sankey["color label"].unique()
        else:
            colors_lab_names = [region + "  "] + data_sankey[
                "color label"
            ].unique().tolist()

    elif dict_col_level[sankey_type] == "0. imp reg":
        labs = data_sankey["color label"].unique().tolist()
        if (region + " ") in labs:
            labs.remove(region + " ")
            colors_lab_names = [region + " "] + labs
        else:
            colors_lab_names = labs
    else:
        colors_lab_names = data_sankey["color label"].unique()
    color_dict = dict(zip(colors_lab_names, colors))
    data_sankey["color"] = data_sankey["color label"].replace(color_dict)
    data_sankey = pd.DataFrame(data_sankey.reset_index())[
        ["source", "target", "color", "value", "position"]
    ]
    data_sankey.set_index(["source", "target", "color", "position"], inplace=True)
    data_sankey[data_sankey < 0] = 0  # this sets to 0 a few negative inventory flows
    data_sankey = data_sankey.groupby(
        level=["source", "target", "color", "position"]
    ).sum()
    data_sankey.reset_index(col_level=0, inplace=True)

    return data_sankey


def data_Sankey(position, sankey_type, region, year, sankey_subtype, version=59):
    """Transforms SLY into three files used to build sankey.

    Parameters
    ----------
    position
    sankey_type : str
        "Commodity", "All commodities", "Commodity all ownership"
    region : str
        region acronym
    year : int
        year
    sankey_subtype : str
        commodity name
    version : int
        version of GLORIA

    Returns
    -------
    node_dict : dictionnary
        dictionnary between nodes names and numbers
    node_list : list
        list of the names of all the nodes
    data_sankey : pd.DataFrame
        all the data needed to build sankey

    """

    if sankey_type == "Commodity all ownership":
        data = feather.read_feather(
            gloria_path
            + "SLY_"
            + str(59)
            + "/"
            + region
            + "/"
            + sankey_subtype
            + "_ownership_"
            + str(year)
            + ".feather"
        )

    elif sankey_type == "Commodity":
        data = feather.read_feather(
            gloria_path + "SLY_" + str(59) + "/" + region + "/" + str(year) + ".feather"
        )

    elif sankey_type == "All commodities":
        data = feather.read_feather(
            gloria_path + "SLY_" + str(59) + "/" + region + "/" + str(year) + ".feather"
        )
    if len(data) == 0:
        return None, None, None
        # print(f"ERROR: {region} {year} {sankey_type} {sankey_subtype}")
        # this is the case for SDS and DYE that are null for some years

    if sankey_type == "Commodity":
        if sankey_subtype == "Aluminium ores":
            data = data.loc[
                ["Bauxite and other aluminium ores concentrates and compounds"]
            ]
        elif sankey_subtype == "Platinum ores":
            data = data.loc[["Platinum group metal ores concentrates and compounds"]]
        elif sankey_subtype == "Other metal ores":
            data = data.loc[
                ["Other metal ores concentrates and compounds nec. including mixed"]
            ]
        else:
            data = data.loc[[sankey_subtype + " concentrates and compounds"]]

    data = add_steps(data, region, sankey_type)

    node_list = []
    for j in data.drop("value", axis=1).columns:
        node_list.extend(list(dict.fromkeys(data.reset_index()[j])))

    node_dict = dict(zip(node_list, list(range(0, len(node_list), 1))))

    data_sankey = data_STV(data, region, node_dict, position, sankey_type)

    return node_dict, node_list, data_sankey


###### functions to run independently


def nodes_data(year, sankey_type, sankey_subtype=None, version=59):
    """For every year and region, saves the files nodes.feather, nodelist.feather and data.feather."""

    position, save_path = variables1(sankey_type, sankey_subtype)
    os.makedirs(save_path, exist_ok=True)

    regions2 = [region for region in region_acronyms.tolist()]

    def process_region_year(region, year):
        save_path2 = os.path.join(save_path, region)
        os.makedirs(save_path2, exist_ok=True)

        node_dict, node_list, data_sankey = data_Sankey(
            position, sankey_type, region, year, sankey_subtype
        )
        if node_dict is None:
            return None
        # this is the case for SDS and DYE that are null for some years

        nodes = pd.DataFrame(
            index=node_list,
            columns=["label kt", "value t", "label t/cap", "value t/cap", "position"],
        )

        target_data = data_sankey.groupby("target").sum()
        source_data = data_sankey.groupby("source").sum()

        for node in node_list:
            try:
                if node_dict[node] in source_data.index:
                    nodes.loc[node, "value t"] = source_data["value"].loc[
                        node_dict[node]
                    ]

                    pos = data_sankey.set_index("source")["position"].loc[
                        node_dict[node]
                    ]
                    nodes.loc[node, "position"] = (
                        pos if isinstance(pos, str) else pos.values[0]
                    )

                else:  # target
                    nodes.loc[node, "value t"] = target_data.loc[
                        node_dict[node], "value"
                    ]
                    nodes.loc[node, "position"] = position[-1]

            except KeyError:
                nodes = nodes.drop(node)
                print(f"ERROR: {node}")

        nodes = nodes[nodes["value t"] > 0]  # Filter out nodes with no value

        dictnodes = {
            "Furnishings, household equipment and routine household maintenance": "Furnishings, household equipment [...]",
            "RoW - Furnishings, household equipment and routine household maintenance": "RoW - Furnishings, household equipment [...]",
            "Iron ores concentrates and compounds": "Iron ores",
            "Silver ores concentrates and compounds": "Silver ores",
            "Bauxite and other aluminium ores - gross ore": "Bauxite and other aluminium ores",
            "Gold ores concentrates and compounds": "Gold ores",
            "Chromium ores concentrates and compounds": "Chromium ores",
            "Copper ores concentrates and compounds": "Copper ores",
            "Manganese ores concentrates and compounds": "Manganese ores",
            "Nickel ores concentrates and compounds": "Nickel ores",
            "Lead ores concentrates and compounds": "Lead ores",
            "Platinum group metal ores concentrates and compounds": "Platinum group metal ores",
            "Tin ores concentrates and compounds": "Tin ores",
            "Titanium ores concentrates and compounds": "Titanium ores",
            "Uranium ores concentrates and compounds": "Uranium ores",
            "Zinc ores concentrates and compounds": "Zinc ores",
            "Other metal ores concentrates and compounds nec. including mixed": "Other metal ores",
        }

        nodes["label kt"] = (
            nodes.rename(index=dictnodes).rename(dictreg).index
            + " ("
            + (nodes["value t"] / 1000).round().astype(int).astype(str)
            + " kt)"
        )

        nodes["label t/cap"] = (
            nodes.rename(index=dictnodes).rename(dictreg).index
            + " ("
            + (
                (nodes["value t"] / pop.loc[region, year]).apply(
                    lambda x: (
                        str(int(x))
                        if x >= 10
                        else (f"{x:.1f}" if 10 > x >= 1 else f"{x:.2f}")
                    )
                )
            )
            + " t/cap)"
        )

        ext = f"{region}{year}.feather"
        feather.write_feather(nodes, os.path.join(save_path2, f"nodes{ext}"))
        feather.write_feather(data_sankey, os.path.join(save_path2, f"data{ext}"))
        feather.write_feather(
            pd.DataFrame(node_list), os.path.join(save_path2, f"nodelist{ext}")
        )

    # Parallelize processing for each region and year
    Parallel(n_jobs=-1)(
        delayed(process_region_year)(region, year) for region in regions2
    )
