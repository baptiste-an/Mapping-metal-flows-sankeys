"""
This module reads GLORIA matrices from zip files csv, and saves them as parquet files.
Parquet is used for efficient storage and retrieval of data.

You should start by downloading the GLORIA data from https://ielab.info/, 
and load these files into the folder specified in the gloria_path variable.

The files used here are e.g. GLORIA_SatelliteAccounts_059_2022.zip, GLORIA_MRIOs_059_2022.zip, 
for every year from 2000 to 2022 and version 59.

After having read the relevant csv files and saved them in a more compact format,
the zip files can be deleted to save space.
"""

import pandas as pd
import zipfile
import os
from itertools import product
import pymrio
import numpy as np

# Path to the GLORIA dataset

############ CHANGE TO THE PATH WHERE YOU SAVED THE GLORIA ZIP FILES ############
gloria_path = "C:/Users/bapti/DATA/gloria/"
#################################################################################

# Version of the GLORIA data
version = 59

# Load region acronyms from the GLORIA ReadMe Excel file
regions = pd.read_excel(
    gloria_path + "GLORIA_ReadMe_0" + str(version) + ".xlsx", sheet_name="Regions"
)
region_acronyms = regions["Region_acronyms"].values

# Load sector names from the GLORIA ReadMe Excel file
sectors = pd.read_excel(
    gloria_path + "GLORIA_ReadMe_0" + str(version) + ".xlsx", sheet_name="Sectors"
)
sector_names = sectors["Sector_names"].values

# Load satellite indicators from the GLORIA ReadMe Excel file
satellites = pd.read_excel(
    gloria_path + "GLORIA_ReadMe_0" + str(version) + ".xlsx", sheet_name="Satellites"
)[["Sat_head_indicator", "Sat_indicator"]]

# Create a multi-index for regions and sectors
multi_index = pd.MultiIndex.from_frame(
    pd.DataFrame(
        list(product(region_acronyms, sector_names)),
        columns=["Region", "Sector"],
    )
)


def date_dict(year, version, file=None):
    """
    Function to determine the correct date for different versions and years.
    This is used to select the correct data files.
    """
    if version == 57:
        datedict_sat = 20231117
        if year in [2021, 2022]:
            datedict = 20230315
    elif version == 59:
        datedict_sat = 20231117
        if year in [i for i in range(1990, 1997, 1)] + [2021, 2022]:
            datedict = 20240110
        elif year in [i for i in range(1998, 2006, 1)] + [
            i for i in range(2007, 2021, 1)
        ]:
            datedict = 20240111
        elif year == 1997:
            if file == "T1":
                datedict = 20240111
            else:
                datedict = 20240112
        elif year == 2006:
            if file in ["V1", "Y5"]:
                datedict = 20240111
            else:
                datedict = 20240110
    return datedict, datedict_sat


def ensure_directory_exists(path):
    """
    Ensure that the directory exists; if not, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def Y(year, version=59):
    """
    Process final demand (Y) data from the GLORIA dataset.
    Extracts data from a zip file, processes it, and saves it in Parquet format.
    """
    zip_file_path = (
        gloria_path + "GLORIA_MRIOs_" + str(version) + "_" + str(year) + ".zip"
    )
    ensure_directory_exists(zip_file_path[:-4])
    date = date_dict(year, version)[0]

    csv_file_name = f"{date}_120secMother_AllCountries_002_Y-Results_{year}_0{version}_Markup001(full).csv"

    # Define final demand categories
    Final_demand_names = [
        "Household final consumption P.3h",
        "Non-profit institutions serving households P.3n",
        "Government final consumption P.3g",
        "Gross fixed capital formation P.51",
        "Changes in inventories P.52",
        "Acquisitions less disposals of valuables P.53",
    ]

    # Create multi-index for final demand data
    index_combinations = list(product(region_acronyms, Final_demand_names))
    multi_index_fd = pd.MultiIndex.from_tuples(
        index_combinations, names=["Region", "Final Demand"]
    )
    # Rows to skip in the CSV file to speed up loading
    skip_rows = [
        i for j in range(0, int(39360 / 240)) for i in range(j * 240, j * 240 + 120)
    ]

    with zipfile.ZipFile(zip_file_path, "r") as z:
        # Read and process the final demand CSV
        Y = pd.read_csv(
            z.open(csv_file_name),
            header=None,
            skiprows=skip_rows,
        )
        Y.index = multi_index
        Y.columns = multi_index_fd
    # Save the processed data as a Parquet file
    Y.to_parquet(zip_file_path[:-4] + "/" + csv_file_name[:-3] + "parquet")


def T(year, version=59):  # 8 min
    """
    Process the transaction matrix (T) data from the GLORIA dataset.
    Extracts data from a zip file, processes it, and saves it in Parquet format.
    """
    zip_file_path = (
        gloria_path + "GLORIA_MRIOs_" + str(version) + "_" + str(year) + ".zip"
    )
    unzipped_file_path = zip_file_path[:-4]
    date = date_dict(year, version, "T1")[0]

    csv_file_name = f"{date}_120secMother_AllCountries_002_T-Results_{year}_0{version}_Markup001(full).csv"

    # Rows to skip in the CSV file to speed up loading
    skip_rows = [
        i for j in range(0, int(39360 / 240)) for i in range(j * 240, j * 240 + 120)
    ]
    cols_to_read = skip_rows

    # Create multi-index for the transaction matrix
    index_tuples = []
    for region in region_acronyms:
        for sector in sector_names:
            index_tuples.append((region, sector))

    multi_index_2 = pd.MultiIndex.from_tuples(index_tuples, names=["Region", "Sector"])

    with zipfile.ZipFile(zip_file_path, "r") as z:
        # Read and process the transaction matrix CSV
        file_path = os.path.join(unzipped_file_path, csv_file_name)
        T = pd.read_csv(
            z.open(csv_file_name),
            header=None,
            skiprows=skip_rows,
            usecols=cols_to_read,
        )
        T.columns = multi_index_2
        T.index = multi_index_2
        # Save the processed data as a Parquet file
        T.to_parquet(file_path[:-3] + "parquet")


def L(year, version=59):
    """
    Calculate the Leontief inverse (L) from the GLORIA dataset.
    Uses the transaction matrix (T) and final demand (Y) to calculate and save L.
    """
    zip_file_path = (
        gloria_path + "GLORIA_MRIOs_" + str(version) + "_" + str(year) + ".zip"
    )
    unzipped_file_path = zip_file_path[:-4]
    date = date_dict(year, version, "T1")[0]

    T_name = f"{date}_120secMother_AllCountries_002_T-Results_{year}_0{version}_Markup001(full).parquet"
    date = date_dict(year, version)[0]
    Y_name = f"{date}_120secMother_AllCountries_002_Y-Results_{year}_0{version}_Markup001(full).parquet"
    L_name = f"{date}_120secMother_AllCountries_002_L-Results_{year}_0{version}_Markup001(full).parquet"
    x_name = f"{date}_120secMother_AllCountries_002_x-Results_{year}_0{version}_Markup001(full).parquet"

    # Load transaction matrix (T) and final demand (Y)
    T = pd.read_parquet(unzipped_file_path + "/" + T_name)
    Y = pd.read_parquet(unzipped_file_path + "/" + Y_name)
    Ymir = Y.clip(lower=0)  # Ensure no negative values

    # Calculate total output (x)
    x = pymrio.calc_x(T, Ymir)
    x.to_parquet(unzipped_file_path + "/" + x_name)

    # Calculate technical coefficient matrix (A)
    A = pymrio.calc_A(T, x)

    # Calculate Leontief inverse (L)
    L = pymrio.calc_L(A)
    # Save the Leontief inverse as a Parquet file
    L.to_parquet(unzipped_file_path + "/" + L_name)


def TQ(year, version=59):
    """
    Process the TQ data (satellite accounts) from the GLORIA dataset.
    Extracts data from a zip file, processes it, and saves it in Parquet format.
    """
    zip_file_path_sat = (
        gloria_path
        + "GLORIA_SatelliteAccounts_0"
        + str(version)
        + "_"
        + str(year)
        + ".zip"
    )
    ensure_directory_exists(zip_file_path_sat[:-4])
    unzipped_file_path_sat = zip_file_path_sat[:-4]
    date_sat = date_dict(year, version)[1]

    csv_file_name = f"{date_sat}_120secMother_AllCountries_002_TQ-Results_{year}_0{version}_Markup001(full).csv"

    # Columns to read from the CSV file to speed up loading
    cols_to_read = [
        i for j in range(0, int(1880 / 240)) for i in range(j * 240, j * 240 + 120)
    ]

    with zipfile.ZipFile(zip_file_path_sat, "r") as z:
        # Read and process the TQ CSV
        TQ = pd.read_csv(
            z.open(csv_file_name),
            header=None,
            usecols=cols_to_read,
        )
        TQ.columns = multi_index
        TQ.index = satellites.apply(tuple, axis=1)
        # Save the processed data as a Parquet file
        TQ.to_parquet(unzipped_file_path_sat + "/" + csv_file_name[:-3] + "parquet")


def S(year, version=59):
    """
    Calculate and save the S matrix.

    This function loads the x and TQ data, calculates the S matrix by dividing
    TQ by the 'indout' column of x, and saves the resulting matrix as a parquet file.
    """
    # Define paths to the unzipped MRIO and satellite account files for the specified year and version
    unzipped_file_path = (
        gloria_path + "GLORIA_MRIOs_" + str(version) + "_" + str(year) + ".zip"
    )[:-4]
    unzipped_file_path_sat = (
        gloria_path
        + "GLORIA_SatelliteAccounts_0"
        + str(version)
        + "_"
        + str(year)
        + ".zip"
    )[:-4]

    # Retrieve dates for MRIO and satellite accounts
    date = date_dict(year, version)[0]
    date_sat = date_dict(year, version)[1]

    # Construct file names for the x, TQ, and S results
    x_name = f"{date}_120secMother_AllCountries_002_x-Results_{year}_0{version}_Markup001(full).parquet"
    TQ_name = f"{date_sat}_120secMother_AllCountries_002_TQ-Results_{year}_0{version}_Markup001(full).parquet"
    S_name = f"{date_sat}_120secMother_AllCountries_002_S-Results_{year}_0{version}_Markup001(full).parquet"

    # Load the 'x' data, replacing zeros with NaN to avoid division errors
    x = pd.read_parquet(unzipped_file_path + "/" + x_name).replace(0, np.nan)

    # Load the TQ data
    TQ = pd.read_parquet(unzipped_file_path_sat + "/" + TQ_name)

    # Calculate S by dividing TQ by the 'indout' column of x
    S = TQ.div(x["indout"], axis=1)

    # Save the resulting S dataframe to a parquet file
    S.to_parquet(unzipped_file_path_sat + "/" + S_name)


def YQ(year, version=59):
    """
    Create and save the YQ matrix.

    This function reads the YQ results from a CSV file in a zip archive,
    assigns appropriate multi-index column names, and saves the data as a parquet file.
    """
    # Define the path to the satellite account zip file for the specified year and version
    zip_file_path_sat = (
        gloria_path
        + "GLORIA_SatelliteAccounts_0"
        + str(version)
        + "_"
        + str(year)
        + ".zip"
    )

    # Retrieve the date for the satellite account
    date_sat = date_dict(year, version)[1]

    # Construct the CSV file name for YQ results
    csv_file_name = f"{date_sat}_120secMother_AllCountries_002_YQ-Results_{year}_0{version}_Markup001(full).csv"

    # Define the names of the final demand categories
    Final_demand_names = [
        "Household final consumption P.3h",
        "Non-profit institutions serving households P.3n",
        "Government final consumption P.3g",
        "Gross fixed capital formation P.51",
        "Changes in inventories P.52",
        "Acquisitions less disposals of valuables P.53",
    ]

    # Create a multi-index for the columns from combinations of region acronyms and final demand names
    index_combinations = list(product(region_acronyms, Final_demand_names))
    multi_index_fd = pd.MultiIndex.from_tuples(
        index_combinations, names=["Region", "Final Demand"]
    )

    # Construct the file name for the parquet file
    YQ_name = f"{date_sat}_120secMother_AllCountries_002_YQ-Results_{year}_0{version}_Markup001(full).parquet"

    # Read the CSV file from the zip archive into a DataFrame
    with zipfile.ZipFile(zip_file_path_sat, "r") as z:
        df = pd.read_csv(z.open(csv_file_name), header=None)

    # Assign multi-index column names to the DataFrame
    df.columns = multi_index_fd
    df.index = pd.MultiIndex.from_frame(satellites)

    # Save the DataFrame to a parquet file
    YQ = df
    YQ.to_parquet(zip_file_path_sat[:-4] + "/" + YQ_name)


def value_added(year, version=59):
    """
    Create and save the value-added matrix.

    This function reads the value-added results from a CSV file in a zip archive,
    assigns appropriate multi-index column names, and saves the data as a parquet file.
    """
    # Define the path to the MRIO zip file for the specified year and version
    zip_file_path = (
        gloria_path + "GLORIA_MRIOs_" + str(version) + "_" + str(year) + ".zip"
    )

    # Retrieve the date for value-added results
    date = date_dict(year, version, "V1")[0]

    # Construct the CSV file name for value-added results
    csv_file_name = f"{date}_120secMother_AllCountries_002_V-Results_{year}_0{version}_Markup001(full).csv"

    # Define the names of the value-added categories
    value_added_names = [
        "Compensation of employees D.1",
        "Taxes on production D.29",
        "Subsidies on production D.39",
        "Net operating surplus B.2n",
        "Net mixed income B.3n",
        "Consumption of fixed capital K.1",
    ]

    # Create a multi-index for the rows from combinations of region acronyms and value-added names
    index_combinations = list(product(region_acronyms, value_added_names))
    multi_index_va = pd.MultiIndex.from_tuples(
        index_combinations, names=["Region", "Final Demand"]
    )

    # Determine the columns to read from the CSV file
    cols_to_read = [
        i for j in range(0, int(39360 / 240)) for i in range(j * 240, j * 240 + 120)
    ]

    # Read the specified columns from the CSV file in the zip archive into a DataFrame
    with zipfile.ZipFile(zip_file_path, "r") as z:
        df = pd.read_csv(z.open(csv_file_name), header=None, usecols=cols_to_read)

    # Assign multi-index column names to the DataFrame
    df.columns = multi_index

    # Assign multi-index row names to the DataFrame
    df.index = multi_index_va

    # Save the DataFrame to a parquet file
    df.to_parquet(zip_file_path[:-4] + "/" + csv_file_name[:-3] + "parquet")
