import re
import shutil
import magic
import pandas as pd
from zipfile import ZipFile
from googletrans import Translator
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from requests import get
import plotly
import plotly.express as px
import plotly.graph_objects as go
from config.config_file import *


def get_location_image_from_google_maps(city_label):
    """
    Google Maps Image scrapper which use Selenium in order to make a specific browser screenshot
    :param city_label:
    :return:
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(chromedriver_path, chrome_options=chrome_options)
    driver.set_window_size(1620, 1200)
    driver.get('https://www.google.com/search?q={}/'.format(city_label))
    img_element = driver.find_element_by_id('lu_map')
    img_location = img_element.location
    img_size = img_element.size
    page_screen = driver.get_screenshot_as_png()
    driver.quit()
    img_read = Image.open(BytesIO(page_screen))
    left, top = img_location['x'], img_location['y']
    right = img_location['x'] + img_size['width']
    bottom = img_location['y'] + img_size['height']
    img_cropped = img_read.crop((left, top, right, bottom))  # defines crop points
    img_cropped.save(img_path + '{}_google_maps.png'.format(city_label))


def get_municipal_population_count(city_label):
    """
    Wikipedia scrapper
    :param city_label:
    :return:
    """
    hp = BeautifulSoup(get('https://fr.wikipedia.org/wiki/{}'.format(city_label)).text, 'html.parser')
    municipal_pop = [a.find_next('td').text.split('hab.')[0] for a in hp.find_all('th') if 'municipale' in a.text][0]
    return municipal_pop


def build_polling_stations_histogram(geocoded_dataframe, city_label):
    """

    :param geocoded_dataframe:
    :param city_label:
    :return:
    """
    # Aggregate by polling station id
    gp = geocoded_dataframe.groupby('place_uid').count()
    gp.reset_index(inplace=True)
    gp.rename(columns={'addresses': 'Total addresses', 'place_uid': 'Polling station number'}, inplace=True)
    gp['Polling station number'] = [str(n + 1) for n in range(gp.shape[0])]
    fig = px.bar(gp,
                 x="Polling station number",
                 y='Total addresses',
                 color='Total addresses',
                 color_continuous_scale='emrld')  # darkmint
    plotly.offline.plot(fig,
                        filename=graph_path+'{}_polling_stations_histogram.html'.format(city_label),
                        auto_open=False)


def build_geocoding_ratio_donut_chart(french_geocoder_instance, city_label):
    """

    :param french_geocoder_instance:
    :param city_label:
    :return:
    """
    total_geocoded_addresses = french_geocoder_instance.total_addresses_from_list
    total_ungeocoded_addresses = french_geocoder_instance.total_ungeocoded_addresses_from_list
    labels = ['Geocoded addresses', 'Ungeocoded addresses']
    values = [total_geocoded_addresses, total_ungeocoded_addresses]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.8, pull=[0.015, 0])])
    fig.update_layout(title_text="% geocoded addresses from municipal list", width=500, height=450)  # 300 / 300
    plotly.offline.plot(fig, filename=graph_path+'{}_geocoding_ratio_donut.html'.format(city_label), auto_open=False)


def detect_file_encoding(file_read):
    """

    :param file_read:
    :return:
    """
    m = magic.Magic(mime_encoding=True)
    encoding = m.from_buffer(file_read)
    return encoding


def detect_file_delimiter(file_stream_decoded_as_str):
    """

    :param file_stream_decoded_as_str:
    :return:
    """
    delimiters = [',', ';', '\t']
    delimiters_dict = {delimiter: file_stream_decoded_as_str.count(delimiter) for delimiter in delimiters}
    delimiter_detected = max(delimiters_dict, key=delimiters_dict.get)
    return delimiter_detected


def unzip_cities_files(input_zip):
    """
    Function which read zip file in memory by extracting all cities files in a dictionary.
    This function auto-detect files extensions, encodings & delimiters

    :param input_zip:
    """

    zip_file = ZipFile(input_zip)
    zip_info = zip_file.infolist()
    cities_files = {}
    for city in zip_info:
        read_input = zip_file.read(city.filename)
        city_label, file_extension = city.filename.split('.')
        city_label = city_label.capitalize()
        encoding_detected = detect_file_encoding(read_input)
        # print(encoding_detected)
        delimiter_detected = detect_file_delimiter(read_input.decode(encoding_detected))
        if file_extension in ['csv', 'txt']:
            cities_files[city_label] = pd.read_csv(BytesIO(read_input), sep=delimiter_detected).reset_index()
        elif file_extension in ['xls', 'xlsx']:
            cities_files[city_label] = pd.read_excel(BytesIO(read_input), sep=delimiter_detected).reset_index()
        else:
            assert "Unvalide file extension"
    return cities_files


def get_coordinates_columns_labels(geocoded_dataframe):
    """
    Function which scope columns that contains coordinates as latitudes and longitudes, by using regex patterns list
    :return: latitudes & longitudes columns labels
    """
    # Regex list which compile main latitude & longitude patterns. Furthermore it will ignore case sensitivity.
    regexes = [
        re.compile('[xy]', re.IGNORECASE),
        re.compile('(lat)', re.IGNORECASE),
        re.compile('(lon[g]?)', re.IGNORECASE)
    ]
    # Get geocoded dataframe columns labels as list
    main_cols = list(geocoded_dataframe.columns)
    # Improve regex (cf GermanGeocoder class)
    coordinates_columns_labels = [x for x in main_cols if any(regex.match(x) for regex in regexes) is True]
    return coordinates_columns_labels


def words_translator(word, language='en'):
    """
    Function which translate a word to another language (default translated language is english) by using
    Google Translate API
    :param word: a word (str)
    :param language: destination language code
    (cf : https://py-googletrans.readthedocs.io/en/latest/#googletrans-languages)
    :return: a word translated
    """
    return Translator().translate(word, dest=language).text


def delete_session_data():
    """
    Function which delete all session data depending on config variable
    """
    if DELETE_ALL_FILES:
        for folder in [csv_path, img_path, graph_path, maps_data_path]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except OSError as oe:
                    print('Failed to delete {}. Error : {}'.format(file_path, oe))


def improve_number_readability(number):
    """
    (Example: 2500 > '2 500')
    :param number: int number
    :return: readable number as str
    """
    str_number = str(number)
    str_number, frag_number = str_number[:-3], str_number[-3:]
    if len(str_number) == 0:
        return frag_number
    else:
        return improve_number_readability(str_number) + ' ' + frag_number


def write_run_time(time_value):
    """

    :param time_value: (str or timedelta object)
    :return:
    """
    time_str = str(time_value)
    time_labels, time_values = ['h', 'min', 's'], time_str.split(':')
    time_dict = {time_label: round(float(time_val)) for time_label, time_val in zip(time_labels, time_values)}
    run_time_str = ' '.join(['{} {}'.format(time_val, k) for k, time_val in time_dict.items() if time_val != 0])
    return run_time_str
