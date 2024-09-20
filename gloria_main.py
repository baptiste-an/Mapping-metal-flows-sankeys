from gloria_preprocessing_initial import *
from gloria_preprocessing_for_sankey import *
from gloria_sankeys import *

"""This scripts contains the four main steps of data processing."""

############_GLORIA_PREPROCESSING_############
"""This extracts the GLORIA matrices from zip files (Y,T,TQ) and calculates the matrices that are missing (L,S).
It then calculates S*L*Y and aggregates it using the concordance table in "DATA/concordance_59.xlsx. """

for year in range(2000, 2023, 1):
    Y(year)
    print("Y")
    T(year)
    print("T")
    L(year)
    print("L")
    TQ(year)
    print("TQ")
    S(year)
    print("S")
    YQ(year)
    value_added(year)
    print(year)
    SLY_agg(year, version=59)
    print("SLY_agg" + str(year))
    SLY_agg_reg(year, version=59)
    print("SLY_agg_reg" + str(year))

############_NODES_DATA_PREPROCESSING_############
"""This transforms the SLY_agg matrices into the format required for the Sankey diagrams.
Nodes data saves all the relevant files in the folders 'Results/Commodity', 'Results/All_commodities' or 'Results/commodity all ownership'.
For every year, sankey type and sankey subtype, three files are saved:
-data, which contains all the data used for the sankeys
-nodelist, a list of the unique names of all the nodes
-nodes, a dataframe with labels, values and position of all nodes. """

for year in range(2000, 2023, 1):
    sankey_type = "All commodities"
    sankey_subtype = None
    nodes_data(year, sankey_type, sankey_subtype, version=59)
    print("nodes_data_" + str(year) + " all commodities")
    for sankey_subtype in [
        "Iron ores",
        "Copper ores",
        "Nickel ores",
        "Lead ores",
        "Zinc ores",
        "Tin ores",
        "Manganese ores",
        "Uranium ores",
        "Gold ores",
        "Silver ores",
        "Aluminium ores",
    ]:
        sankey_type = "Commodity all ownership"
        nodes_data(year, sankey_type, sankey_subtype, version=59)
        print("nodes_data_" + str(year) + "_" + sankey_subtype + " ownership")
        sankey_type = "Commodity"
        nodes_data(year, sankey_type, sankey_subtype, version=59)
        print("nodes_data_" + str(year) + "_" + sankey_subtype)
    # remaining commodities
    for sankey_subtype in [
        "Chromium ores",
        "Platinum ores",
        "Titanium ores",
        "Other metal ores",
    ]:
        sankey_type = "Commodity"
        nodes_data(year, sankey_type, sankey_subtype, version=59)
        print("nodes_data_" + str(year) + "_" + sankey_subtype)


############_SANKEY_PREPROCESSING_############
"""This transforms the nodes/nodelist/data files into a single .pkl.lmza file.
This .pkl.lzma file contains all the info required for the sankeys in a format appropriate for go.Sankey."""

regions2 = region_acronyms.tolist()
ensure_directory_exists("Results/Sankey_preprocessed/")
for year in range(2000, 2023, 1):
    ensure_directory_exists("Results/Sankey_preprocessed/All commodities")
    for region in regions2:
        preprocess_sankey_data(year, region, "All commodities", None)
    for sankey_subtype in [
        "Iron ores",
        "Copper ores",
        "Nickel ores",
        "Lead ores",
        "Zinc ores",
        "Tin ores",
        "Manganese ores",
        "Uranium ores",
        "Gold ores",
        "Silver ores",
        "Aluminium ores",
    ]:
        ensure_directory_exists(
            "Results/Sankey_preprocessed/Commodity all ownership/" + sankey_subtype
        )
        ensure_directory_exists(
            "Results/Sankey_preprocessed/Commodity/" + sankey_subtype
        )
        for region in regions2:
            preprocess_sankey_data(
                year, region, "Commodity all ownership", sankey_subtype
            )
            preprocess_sankey_data(year, region, "Commodity", sankey_subtype)

    for sankey_subtype in [
        "Chromium ores",
        "Platinum ores",
        "Titanium ores",
        "Other metal ores",
    ]:
        ensure_directory_exists(
            "Results/Sankey_preprocessed/Commodity/" + sankey_subtype
        )
        for region in regions2:
            preprocess_sankey_data(year, region, "Commodity", sankey_subtype)

############_SANKEY_PLOTTING_############

"""This creates the Sankey diagrams for all the years, regions and commodities.
It saves the sankeys as .svg files in the folder 'Results/Sankey figs'."""

save_path = "Results/Sankey figs/"
ensure_directory_exists(save_path)
sankey_type = "All commodities"
ensure_directory_exists(save_path + sankey_type)
sankey_subtype = None
for region in region_acronyms:
    for year in range(2001, 2023, 1):
        for unit2 in ["kt", "tcap"]:
            if unit2 == "tcap":
                unit = "t/cap"
            else:
                unit = "kt"

            try:
                fig = fig_sankey(region, year, unit, sankey_type, sankey_subtype)
                ensure_directory_exists(save_path + sankey_type + "/" + dictreg[region])
                fig.write_image(
                    save_path + sankey_type + "/" + dictreg[region] + "/" + sankey_type
                    # + "_"
                    # + sankey_subtype
                    + "_" + region + "_" + str(year) + "_" + unit2 + ".svg",
                    engine="orca",
                )
            except FileNotFoundError:
                print(region + str(year))


for sankey_type in ["Commodity", "Commodity all ownership"]:
    ensure_directory_exists(save_path + sankey_type)
    for sankey_subtype in [
        "Iron ores",
        "Copper ores",
        "Nickel ores",
        "Lead ores",
        "Zinc ores",
        "Tin ores",
        "Manganese ores",
        "Uranium ores",
        "Gold ores",
        "Silver ores",
        "Aluminium ores",
    ]:
        ensure_directory_exists(save_path + sankey_type + "/" + sankey_subtype)
        for region in region_acronyms:
            for year in range(2001, 2023, 1):
                for unit2 in ["kt", "tcap"]:
                    if unit2 == "tcap":
                        unit = "t/cap"
                    else:
                        unit = "kt"
                    try:
                        fig = fig_sankey(
                            region, year, unit, sankey_type, sankey_subtype
                        )
                        ensure_directory_exists(
                            save_path
                            + sankey_type
                            + "/"
                            + sankey_subtype
                            + "/"
                            + dictreg[region]
                        )
                        fig.write_image(
                            save_path
                            + sankey_type
                            + "/"
                            + sankey_subtype
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
                            + "_"
                            + unit2
                            + ".svg",
                            engine="orca",
                        )
                    except FileNotFoundError:
                        print(region + str(year))

sankey_type = "Commodity"
ensure_directory_exists(save_path + sankey_type)
for sankey_subtype in [
    "Chromium ores",
    "Platinum ores",
    "Titanium ores",
    "Other metal ores",
]:
    ensure_directory_exists(save_path + sankey_type + "/" + sankey_subtype)
    for region in region_acronyms:
        for year in range(2001, 2023, 1):
            for unit2 in ["kt", "tcap"]:
                if unit2 == "tcap":
                    unit = "t/cap"
                else:
                    unit = "kt"
                try:
                    fig = fig_sankey(region, year, unit, sankey_type, sankey_subtype)
                    ensure_directory_exists(
                        save_path
                        + sankey_type
                        + "/"
                        + sankey_subtype
                        + "/"
                        + dictreg[region]
                    )
                    fig.write_image(
                        save_path
                        + sankey_type
                        + "/"
                        + sankey_subtype
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
                        + "_"
                        + unit2
                        + ".svg",
                        engine="orca",
                    )
                except FileNotFoundError:
                    print(region + str(year))
