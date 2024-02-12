import datetime
import flask_monitoringdashboard as dashboard
from flask import Flask
from celery import Celery
from geomodules.french_geocoder.src.polling_stations_french_geocoder import FrenchGeocoder
from geomodules.cartographer.src.cartography_outlines import PSOCartographer
from utils.psomap_utilities import *
from utils.custom_urls_converters import ListConverter
from config.config_file import tasks_result_main_dict

# Manage settings warnings copying
pd.options.mode.chained_assignment = None

# Main app configuration
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['SECRET_KEY'] = 'pso'
# Add Flask custom url list type
app.url_map.converters['list'] = ListConverter
# Add Flask monitoring dashboard
dashboard.bind(app)

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@celery.task(bind=True)
def build_tracing_map_task(self, addresses_list_json, city_label):
    """Background task that runs a long function with progress reports."""
    # Get location google maps image
    get_location_image_from_google_maps(city_label)
    # Convert json addresses list to dataframe
    addresses_list_df = pd.read_json(addresses_list_json)
    addresses_list_df.drop(columns='index', inplace=True)
    # Check if addresses list is already geocoded
    coordinates_columns_labels = get_coordinates_columns_labels(addresses_list_df)
    fg = ""
    if len(coordinates_columns_labels) == 2:
        geocoded_df = addresses_list_df
        geocoding_ratio = 100
        geocoding_time_execution = 0
    else:
        # Geocode addresses list
        fg = FrenchGeocoder(addresses_list_df, city_label)
        # Compute geocoding run time
        start_fg = datetime.datetime.now()
        geocoded_df = fg.build_geocoded_csv()
        end_fg = datetime.datetime.now()
        geocoding_time_execution = end_fg - start_fg
        addresses_not_found_df = fg.ungeocoded_df
        # TODO : IMPROVE METHOD (READING MEMORY BASED)
        addresses_not_found_df.to_csv(csv_path+'{}_addresses_not_found.csv'.format(city_label), index=False, sep=';')
        geocoding_ratio = fg.compute_geocoding_ratio()
    # Build donut chart
    build_geocoding_ratio_donut_chart(fg, city_label)
    # Build polling stations histogram
    build_polling_stations_histogram(geocoded_df, city_label)
    # Get municipal code
    municipal_code = addresses_list_df.place_uid.unique().tolist()[0]
    # Get municipal population
    municipal_population = get_municipal_population_count(city_label)
    # Get polling stations count
    polling_stations_count = len(geocoded_df.place_uid.unique().tolist())
    # Total geocoded addresses
    geocoded_addresses_count = geocoded_df.shape[0]
    # Addresses count per polling stations
    geocoded_addresses_count_per_polling_stations = geocoded_addresses_count // polling_stations_count
    # Compute tracing run time
    start_tracing = datetime.datetime.now()
    # Build polling stations outlines
    PSOCartographer(geocoded_df, city_label).build_polling_stations_outlines(ways_type='walk')
    end_tracing = datetime.datetime.now()
    tracing_time_execution = end_tracing - start_tracing
    # Compute total run time
    if geocoding_time_execution == 0:
        total_run_time = str(tracing_time_execution)
    else:
        total_run_time = str(geocoding_time_execution + tracing_time_execution)

    # Improve run time readability
    geocoding_time, tracing_time, total_task_time = [write_run_time(v) for v in [geocoding_time_execution,
                                                                                 tracing_time_execution,
                                                                                 total_run_time]]
    self.update_state(state='PROGRESS')

    # Get all variables which will be added to tasks result dictionary
    tasks_variables = ['success', city_label, municipal_code, municipal_population, geocoded_addresses_count,
                       geocoded_addresses_count_per_polling_stations, polling_stations_count, geocoding_ratio,
                       geocoding_time, tracing_time, total_task_time]

    # Improve number readability for each int tasks variables
    tasks_variables = [improve_number_readability(v) if type(v) is int else v for v in tasks_variables]

    # Update tasks result main dictionary values
    for tasks_result_key, task_variable in zip(tasks_result_main_dict, tasks_variables):
        tasks_result_main_dict[tasks_result_key] = task_variable

    tasks_result_header_dict = {'current': 100, 'total': 100, 'status': '{} completed!'.format(city_label)}
    tasks_result_header_dict.update(tasks_result_main_dict)

    return tasks_result_header_dict


def tracing_map_task_status(tracing_task, city_label):
    """

    """
    if tracing_task.state == 'PENDING':
        response = {
            'state': tracing_task.state,
            'current': 0,
            'total': 1,
            'status': '{} pending...'.format(city_label),
        }
    elif tracing_task.state != 'FAILURE':
        response = {
            'state': tracing_task.state,
            'current': tracing_task.info.get('current', 0),
            'total': tracing_task.info.get('total', 1),
            'status': tracing_task.info.get('status', ''),
        }

        for info_label in tracing_task.info:
            if info_label in tasks_result_main_dict.keys():
                response[info_label] = tracing_task.info[info_label]
    else:
        # something went wrong in the background job
        response = {
            'state': tracing_task.state,
            'current': 1,
            'total': 1,
            'status': str(tracing_task.info) + 'Error',  # this is the exception raised
        }

    return response
