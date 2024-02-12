# French Geocoder


This project is a first version of a french geocoder which will be used by FME algorithm.

It builds a geocoded csv file by merging an adresses list with a regional subset extracted from latest BAN API
from data.gouv. 

This module use a fuzzy matching method which use Levenshtein distance (with a specific decision rule)
in order to match similar addresses.

## Description

This project contains 3 modules which assumed the whole geocoding process :

 - **polling_stations_french_geocoder.py** : a geocoding class which geocode all addresses referred to a particular location
 - **extract_osm_from_pbf.py** : an optional module which extract roads data from Geofabrik URL
 - **french_geocoder_wrappers.py** : wrappers which encapsulate previous modules classes
 
The geocoding script need a unique file in order to geocode french addresses which is :

  - A csv/xls(x) file which contains a french addresses list (streets which will be geocoded)
  
**N.B : data paths management is configured in order to fit with FME project main directory.**

If you want to use this module externally, then you will need to :

  - uncomment this main block in the config file  (and comment back previous paths variables placed before this comment block)
  - run the build_alternatives_data_paths.py module

Step 1 : activate alternatives paths

```

# Alternatives paths : paths variables needed in order to use this module externally

databases_input_path  = 'database/input/'

path_csv_geocoded = 'database/output/'

input_addresses_list_path = input_path + 'addresses_list/'

```

Step 2 : run alternatives paths builder module


Command line :

```
python utils/build_alternatives_data_paths.py 
```

Then you could upload your csv files in their respective folders :

  - addresses databases csv file need to be stored in "addresses_databases_files" folder
  - addresses list csv file need to be stored in "addresses_list" folder

## Prerequisites

Install modules needed written in a requirements.txt file

Command line : 

```
pip install -r requirements.txt
```

N.B : you can auto-generate a requirements.txt file by using the module generate_requirements.py

Command line :

```
python utils/generate_requirements.py 
```

## Getting Started


### Main file example

Run the main.py file in order to start the geocoding example process.

Command line : 

```
python main.py
```

### Jupyter notebook example

Run this script in a notebook with jupyter in order to start the geocoding process.

```
from config.config_file import french_geocoder_params_pattern
from src.polling_stations_french_geocoder import FrenchGeocoder

# Assign a specific location label in french geocoder parameters dictionary
french_geocoder_params_pattern['location_filename'] = 'Colombes'
                             
# Create a FrenchGeocoder instance
french_geocoder_instance = FrenchGeocoder(**french_geocoder_params_pattern)

# Run the geocoding process
french_geocoder_instance.build_geocoded_csv()

# Print the first heading rows of the geocoded dataframe
french_geocoder_instance.geocoded_df.head()
```

