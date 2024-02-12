import os

# Output data paths by output type
csv_path = os.getcwd() + '/static/csv/'
img_path = os.getcwd() + '/static/img/'
graph_path = os.getcwd() + '/static/graph/'
maps_data_path = os.getcwd() + '/static/maps_data/'

# Selenium chromedriver path (google maps image scrapper)
chromedriver_path = os.getcwd() + '/config/chromedriver'

# Enable/disable output data deletion
DELETE_ALL_FILES = True

# Tasks dictionary pattern
tasks_result_main_dict = {'result': "",
                          'city_label': "",
                          'municipal_code': "",
                          'municipal_population': "",
                          'geocoded_addresses_count': "",
                          'geocoded_addresses_count_per_polling_stations': "",
                          'polling_stations_count': "",
                          'geocoding_ratio': "",
                          'geocoding_run_time': "",
                          'tracing_run_time': "",
                          'total_run_time': ""}
