from utils.french_geocoder_wrappers import geofabrik_wrapper, french_geocoder_wrapper
from config.config_file import geofabrik_params_pattern, french_geocoder_params_pattern

# Assign a specific location label (which came from a csv filename stored in the main input french addresses list
# subdirectory related to bdv-fme project) in french geocoder parameters dictionary
french_geocoder_params_pattern['location_filename'] = 'location_name'
# Use french geocoder main class wrapper in order to run an automated geocoding session
fgw_dict = french_geocoder_wrapper(**french_geocoder_params_pattern)
# Print firsts rows from a geocoded dataframe stored in previous result dictionary
fgw_dict['geocoded_dataframe'].head()

# Optional : download roads osm.pbf for a specific location from http://download.geofabrik.de/
# geofabrik_wrapper(**geofabrik_params_pattern)
