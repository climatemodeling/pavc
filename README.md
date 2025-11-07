# Overview: Pan-Arctic Vegetation Cover (PAVC) Database
**Contributors**: [Morgan Steckler](https://github.com/msteckle), [Tianqi Zhang](https://github.com/zhang1206), [Jitendra (Jitu) Kumar](https://github.com/jitendra-kumar)

This repository contains the jupyter notebooks (and supporting python modules) used to synthesize species and plant functional type (PFT) vegetation cover into one comprehensive database: PAVC. Version 1.0 of the database includes plot data from:

- the Arctic Vegetation Archive of Alaska (AVA),
- the Alaska Vegetation (AKVEG) database,
- Alaska Biological Research (ABR) data associated with Macander et al. (2017),
- the National Ecological Observation Network,
- and previously unpublished Seward Peninsula data from the Next Generation Ecosystem Experiments: Arctic (NGA).

# Clone this repository
Use a password-protected ssh key:
```bash
git clone https://github.com/climatemodeling/pavc.git
```
**_How to set up an ssh key:** https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## Dependencies (tested on Linux):
This repository requires the following non-python standard libraries:
- gdal 3.4.2
- openmpi 5.0.5

### Install openmpi (without sudo permissions):
```bash
cd $HOME
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz
tar -xvf openmpi-5.0.5.tar.gz
cd openmpi-5.0.5
INSTALL_PREFIX=$HOME/openmpi-5.0.5-install
./configure --prefix=$INSTALL_PREFIX
make -j8
make install
```

### Install gdal (without sudo permissions):
```bash
cd $HOME
wget https://download.osgeo.org/gdal/3.4.2/gdal-3.4.2.tar.gz
tar -xvf gdal-3.4.2.tar.gz
cd gdal-3.4.2
export OPENMPI_DIR=$HOME/openmpi-5.0.5-install
./configure --prefix=$HOME/gdal-3.4.2-install --with-openmpi=$OPENMPI_DIR
make -j8
make install
```

## Install python dependencies:
We recommend using `uv` to manage your python environment. You can create a new environment with the required dependencies by running:
```bash
uv new -p python=3.9 alaska_pft_fcover_harmonization
uv activate alaska_pft_fcover_harmonization
uv pip install -r requirements.txt
```

# Contribution: How can you improve the database?

These detailed Jupyter notebooks and helper functions were designed to streamline the process of synthesizing new datasets. Once tables have been formatted correctly, a new jupyter notebook can be written to synthesize data according to our standards. Notebooks provide detailed instructions and notes on how the data were pre-processed and synthesized using both automated and manual steps. If you come across issues in the notebooks, modules, or data, please publish an issue in this Github. We also welcome feedback and suggestions. We are always on the hunt for new data, so if you know of a dataset that isn't yet in the database that you'd like to see synthesized, please let us know!

## Issue Tags:
- `new dataset` (suggestion for a new dataset to add to PAVC)
- `bug` (something isn't working or seems incorrect)
- `documentation` (suggestions for improving documentation)
- `duplicate` (this issue or pull request already exists)
- `enhancement` (suggestions for improvement)
- `question` (further information is requested)

<br>

# Generalized Synthesis Steps
If you would like to try synthesizing a dataset for input into our database, take a look at the general procedure outlined below. We suggest following each code block in our Jupyter notebooks. You'll notice the steps are very similar across all notebooks.

---
## Pre-processing procedure:
First, ensure that the dataset you want to synthesize meets the bare minimum requirements for inclusion in the database:
- Cover is measured at the species-level
- Cover is measured as absolute cover, the proportion of the plot's area covered by vegetation spread across all heights in the plant community (can sum to over 100 percent)
- Non-vegetation cover is available as top cover, the proportion of the plot's area covered by vegetation in only the top layer of the plant community (it should always sum to 100 percent)
	- Water
 	- Bare Ground (bare soil, bare ground, rock, etc.)
- Information about the plot's location, survey methods, authors, etc. are accessible

1. **Visually inspect, clean, and format cover tables before standardization**
    - Rows should be species names (without author if possible)
    - Column headers should be _unique_ plot IDs (ensure uniqueness!!!)
    - Cell values should be the cover
    - There should be no extra blank columns or rows, nor extra column/row labels
    - Encoded as 'utf-8-sig' in Python

<br>

2. **Gather plot-level and survey-level auxiliary information**
We categorize our auxiliary information as temporal, spatial, contextual, and geo-contextual. Below are the auxiliary plot information that contributors should assemble when synthesizing a dataset according to PAVC.
![PFT_database_visual (1)](https://github.com/user-attachments/assets/048f8d3a-fb6e-43c5-8264-b57a1eaa6261)

3. **Correct cover values that, when converted to float, raise a data-type error**
    - For example, "Trace" was sometimes used instead of 0.01
    - Human error
    
<br>

4. **If applicable, convert 0.0 cover to a trace value like 0.01.**
Some datasets indicate trace with a small percent, with the text 'Trace', 'T', etc. Some datasets include 0 cover to mean that a species was present in the plot but at very trace amounts. In our database, we standardize trace by assigning a small value like 0.01 to the 0 cover values if applicable.

5. **Ensure cover is represented as a percent**
    - Older plots used Braun-Blanquet or Westoff oridnal codes that indicate a range of cover; the mid-point percentage was used as the representative percent cover value
    - If working with ordinal cover codes, ensure you know the percentage ranges that represent each code---this can vary from project to project.
    
<br>

## Species and PFT standardization procedure:
6. **Join the species names from your cover table to the species name in the AKVEG Species Checklist to identify potential habits and accepted species names**
You'll be happy to know that we wrote a big 'ole function for this in our support python module. You should use that to match your table's species names to the AKVEG checklist.

7. **Join the species names from your cover data to the species name in the Leaf Habit Supplementary Table from Macander et al. (2020) to assign shrubs/trees a leaf retention type (deciduous, evergreen)**

9. **Export the join of species names, their associated "potential" accepted names, and growth habits**
The list of potential accepted names is supposed to aid the user in assigning an accepted name. If there was no match between your species name and an AKVEG species name, you should consult with an expert to assign an accepted species name, author, and naming authority.

11. **Manually ID one accepted species name for each of your dataset's species names**
The manual species name adjudication process is the most time-consuming component of this project. The species names from your data table should be creafully assigned an accepted species name. You also should know the author of the species name, as well as the authority you referenced during accepted name assignment. You'll also need to assign whether the name you've chosen is at the family, genus, species, subspecies, variety, or type (functional type) level.

12. **Aggregate species-level fcover to PFT-level fcover by grouping species into their PFTs and summing**
We have specific PFT categories: lichen and bryophytes (sum = non-vascular), graminoid, deciduous shrub, evergreen shrub, deciduous tree, evergreen tree, and other; as well as litter, all represented as absolute foliar cover. Water and bare ground are also necessary, but they should be measured as top cover.

---
---
