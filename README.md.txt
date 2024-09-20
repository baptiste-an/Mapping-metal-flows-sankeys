# Mapping regional metal flows from mine ownership to final consumption

The goal of this project is to build sankey diagrams to map the value chains of metals. We build over 100,000 Sankey diagrams to visualize metal flows across 159 countries and 4 world regions, covering three key steps: mine owner nationality, extraction region, and final consumption region.

These diagrams are based on the GLORIA MRIO, available at https://ielab.info/, as well as on S&P Global data for mine ownership.

## Requirements

```bash
$ pip install -r requirements.txt
```

## Usage

The first step is to download GLORIA data from https://ielab.info/, and load these files into the folder specified in the gloria_path variable in the gloria_processing_initial.py file.
The files used here are e.g. GLORIA_SatelliteAccounts_059_2022.zip, GLORIA_MRIOs_059_2022.zip, for every year from 2000 to 2022 and version 59. After having read the relevant csv files and saved them in a more compact format, the zip files can be deleted to save space.

```bash
$ python gloria_main.py
```
It:

* Extracts the GLORIA matrices from zip files (Y,T,TQ) and calculate the matrices that are missing (L,S)
* Calculates S*L*Y and aggregates it using the concordance table in "DATA/concordance_59.xlsx
* Transforms the SLY_agg matrices into the format required for the Sankey diagrams.
* Saves all the relevant files in the folders 'Results/Commodity', 'Results/All_commodities' or 'Results/commodity all ownership'.
* Creates the Sankey diagrams for all the years, regions and commodities.
* Saves the sankeys as .svg files in the folder 'Results/Sankey figs'.

## Application

The code to build the application has been adapted from: https://github.com/baptiste-an/Application-mapping-GHG

## Citation

Andrieu, B., Cervantes Barron, K., Heydari, M., Keshavarzzadeh, A., Cullen, J., Mapping regional metal flows from mine ownership to final consumption