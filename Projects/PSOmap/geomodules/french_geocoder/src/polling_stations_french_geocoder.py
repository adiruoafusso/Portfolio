import os
import gzip
import urllib.request
import pandas as pd
import numpy as np
import geocoder
from fuzzywuzzy import fuzz, process
from unidecode import unidecode


class FrenchGeocoder:
    """
    Run a global geo-coding process for a specific location
    CSV files need to be configured as followed :
    - the filename must start with location name (example : "Ivry.csv")
    - the csv file must contains these columns :
      - street : location streets labels
      - polling_station_id : location polling stations numbers
      - number_start : int value which refers to the starting street counting number
      - number_stop : int value which refers to the ending street counting number
      - even_numbers_only : boolean (Yes/No) which specify if a street contains only even numbers
      - odd_numbers_only : boolean (Yes/No) which specify if a street contains only odd numbers
      - city : location label
      - place_uid : location zip code

    """

    def __init__(self, addresses_list_dataframe, location_label):
        """
        :param location_filename: csv location filename (example : 'Ivry') (type: str)
        :param sep: csv separator (type: str)
        :param file_extension: extension's format : '.extension' (type: str)
        """
        ################################################################################################################
        #                                            READING PARAMETERS                                                #
        ################################################################################################################
        # City label capitalized
        self.location_label = location_label.capitalize()  # Type str (Example : "Ivry")
        self.addresses_list_df = addresses_list_dataframe
        # Normalize columns labels
        lowerized_columns_labels_dict = {col: col.lower() for col in self.addresses_list_df.columns.tolist()}
        self.addresses_list_df.rename(columns=lowerized_columns_labels_dict, inplace=True)
        # Optimize data types based on columns labels
        self.optimize_data_types()
        # Create place uid pattern & insee code variables
        self.place_uid_pattern = str(self.addresses_list_df.place_uid.iloc[0])
        self.insee_code_first_two_digits = self.place_uid_pattern[:2]
        # Run time approximation
        self.run_time_ratio = 0
        self.addresses_db_df = self.get_location_addresses_db_from_ban()
        self.terms_translator = ""
        # Output static
        self.addresses_dict_filtered = ""
        self.unknown_addresses = ""
        self.addresses_dict_geocoder = {}
        # Unfounded addresses
        self.addresses_not_found = []
        self.geocoded_df = ""
        self.ungeocoded_df = ""
        # Donut chart data
        self.total_addresses_from_list = ""
        self.total_ungeocoded_addresses_from_list = ""

    ####################################################################################################################
    #                                         Utility Functions (static-cleaning)                                      #
    ####################################################################################################################

    @staticmethod
    def convert_string_to_boolean(value_str):
        """
        Static method which convert a string value as boolean
        :param value_str: a string value which represent a boolean ('Yes', 'No')
        return: a boolean value
        """
        if value_str.lower() == 'yes':
            return True
        else:
            return False

    def optimize_data_types(self):
        """
        Method which optimize data types for each columns labels, by reducing the size of addresses list dataframe
        cf:
        https://medium.com/@vincentteyssier/optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment-5f07db3d72e
        """
        for col in self.addresses_list_df.columns:
            if col in ['even_numbers_only', 'odd_numbers_only']:
                self.addresses_list_df[col] = self.addresses_list_df[col].map(self.convert_string_to_boolean)
                self.addresses_list_df.loc[:, col] = self.addresses_list_df.loc[:, col].astype('bool')
            elif col in ['place_uid', 'city']:
                self.addresses_list_df.loc[:, col] = self.addresses_list_df.loc[:, col].astype('category')
            elif col in self.addresses_list_df.select_dtypes(include=['object']).columns:
                self.addresses_list_df.loc[:, col] = self.addresses_list_df.loc[:, col].astype('str')
            elif col in self.addresses_list_df.select_dtypes(include=['int64']).columns:
                self.addresses_list_df.loc[:, col] = pd.to_numeric(self.addresses_list_df[col], downcast='signed')

    @staticmethod
    def compute_levenshtein_dist(word, words_list, metric=fuzz.token_set_ratio):
        """
        Static method which compute Levenshtein distance between a word and a list of words candidates

        :param word: a specific word (str)
        :param words_list: a list of words (list)
        :param metric: a specific scorer from fuzzywuzzy package
        (scorers list : https://github.com/seatgeek/fuzzywuzzy/blob/master/fuzzywuzzy/fuzz.py)

        :return: a dictionary with the word tested as key and a list of tuples (candidates & ratios) as value
        """
        return process.extract(word, words_list, scorer=metric)

    @staticmethod
    def normalize_encoding_in_address_col(df, col, lowercase=False):
        """
        Static method which normalize addresses encoding (by removing accents & specials characters)

        :param df: a dataframe
        :param col: a column which will be normalized
        :param lowercase: lower a string

        :return: a normalized dataframe
        """
        if lowercase:
            df[col] = [unidecode(u'{0}'.format(accented_str)).lower() for accented_str in df[col]]
        else:
            df[col] = [unidecode(u'{0}'.format(accented_str)) for accented_str in df[col]]
        return df

    def translate_term_in_addresses_col(self, df, col, terms_translator=None):
        """
        Method which map a dictionary to a specific dataframe column by using regex

        :param df: a dataframe
        :param col: a column which will be mapped
        :param terms_translator: a translation dictionary
        """
        if terms_translator is not None:
            self.terms_translator = terms_translator
        terms_keys, terms_values = list(self.terms_translator.keys()), list(self.terms_translator.values())
        df[col].replace(to_replace=terms_keys, value=terms_values, regex=True, inplace=True)
        # Update run time count
        self.run_time_ratio += 5

    def get_location_addresses_db_from_ban(self, delete_csv_file=True):
        """
        Method which extract and build a regional subset from BAN
        :param delete_csv_file: default parameter which delete the main regional csv file extracted from BAN (boolean)
        :return: a dataframe
        """
        # BAN export API URL
        ban_data_gouv_api = 'https://adresse.data.gouv.fr/data/ban/export-api-gestion/latest/ban/'
        #
        ban_db_gz_file = 'ban-{}.csv.gz'.format(self.insee_code_first_two_digits)
        url_requested = ban_data_gouv_api + ban_db_gz_file
        ban_db_csv_file = ban_db_gz_file[:-3]

        response = urllib.request.urlopen(url_requested)
        with open(ban_db_csv_file, 'wb') as outfile:
            outfile.write(gzip.decompress(response.read()))

        location_addresses_db_raw = pd.read_csv(ban_db_csv_file, sep=';')
        if delete_csv_file is True:
            cwd = os.getcwd()
            os.remove(cwd + '/' + ban_db_csv_file)
        location_addresses_db = location_addresses_db_raw[location_addresses_db_raw.nom_commune == self.location_label]
        location_addresses_db_subset = location_addresses_db[['numero', 'suffixe', 'nom_voie', 'lon', 'lat']]
        # Update run time count
        self.run_time_ratio += 5
        return location_addresses_db_subset

    def fuzzy_matching(self, quality_threshold=88):
        """
        Matching method which apply Levenshtein distance computation method in order to match addresses
        (from addresses list related to addresses database list) which have different patterns labels

        N.B : this method do an "extra-mile" by trying to geocode unmatched addresses (which mean aren't contained in
        BAN) with an external API (https://geocoder.readthedocs.io/)
        """
        self.normalize_encoding_in_address_col(self.addresses_list_df, 'street', lowercase=True)
        self.normalize_encoding_in_address_col(self.addresses_db_df, 'nom_voie', lowercase=True)
        # Get uniques addresses list from previous subset (addresses are filtered by non-alphanumerics characters)
        addresses_from_streets_list = set(self.addresses_list_df.street)
        # Get uniques addresses list from addresses database street label column values
        addresses_from_db_subset = set(self.addresses_db_df.nom_voie)
        # Build a translation dictionary which compute Levenshtein distance between theses two sets
        addr_dict = {a: self.compute_levenshtein_dist(a, addresses_from_db_subset)
                     for a in addresses_from_streets_list if a not in addresses_from_db_subset}

        # Filter unknown addresses (addresses not found in database) by applying a quality threshold
        self.addresses_dict_filtered = {k: v[0][0] if (k.split()[-1] in v[0][0] or v[0][1] >= quality_threshold) else "" for k, v in addr_dict.items()}
        # Extract unknown addresses (apply same logic mentioned earlier)
        self.unknown_addresses = [k for k, v in self.addresses_dict_filtered.items() if v == ""]
        # Geocode unmatched addresses
        self.geocode_unmatched_addresses()
        # Update run time count (+ 2 statics methods)
        self.run_time_ratio += 10  # old 15

    def geocode_unmatched_addresses(self):
        """
        Method which try to geocode unmatched addresses with geocoder API
        """
        for address in self.unknown_addresses:
            try:
                geocoded_result = geocoder.osm(address)
                coordinates_result = [geocoded_result.lng, geocoded_result.lat]
                if list(set(coordinates_result))[0] is not None:
                    self.addresses_dict_geocoder[address] = coordinates_result
                else:
                    self.addresses_not_found.append(address)
            except:
                pass
        # Update run time count
        self.run_time_ratio += 5

    def translate_fuzzy_matching(self):
        """
        Apply the translate_term_in_addresses_col method to addresses list dataframe with
        the translation dictionary built from fuzzy matching method
        :return: a translated dataframe
        """
        self.fuzzy_matching()
        # Won't overwrite addresses list dataframe by creating another variable
        addresses_list_matched_df = self.addresses_list_df.copy()
        # Apply the translate_term_in_addresses_col method
        self.translate_term_in_addresses_col(addresses_list_matched_df, 'street', self.addresses_dict_filtered)
        # Return the addresses list dataframe translated
        addresses_list_matched_df.rename(columns={'street': 'nom_voie'}, inplace=True)
        result = addresses_list_matched_df[addresses_list_matched_df.nom_voie != ""][['nom_voie', 'polling_station_id']]
        # Update run time count
        self.run_time_ratio += 5
        return result

    def merge_perfect_match(self):
        """
        Method which perform a perfect matching between addresses with same labels.
        Join type : inner join
        :return: a merged dataframe
        """
        addresses_list_matched_df = self.translate_fuzzy_matching()
        perfect_match_merged_df = addresses_list_matched_df.merge(self.addresses_db_df, on='nom_voie')
        # Update run time count
        self.run_time_ratio += 5
        return perfect_match_merged_df

    def complete_merge(self):
        """
        Method which perform a complete merge operation by adding recalcitrant addresses geocoded with geocoder API
        :return: a merged dataframe
        """
        perfect_match_merged_df = self.merge_perfect_match()
        recalcitrant_addresses = list(self.addresses_dict_geocoder.keys())
        if len(recalcitrant_addresses) == 0:
            complete_merged_df = perfect_match_merged_df
        else:
            addresses_list_df_scoped = self.addresses_list_df[self.addresses_list_df.street.isin(recalcitrant_addresses)]
            addresses_list_df_scoped.loc[:, 'coords'] = addresses_list_df_scoped.street.map(self.addresses_dict_geocoder)
            addresses_list_df_scoped[['lon', 'lat']] = pd.DataFrame(addresses_list_df_scoped.coords.values.tolist(),
                                                                    index=addresses_list_df_scoped.index)
            addresses_list_df_scoped = addresses_list_df_scoped[['street', 'polling_station_id', 'lon', 'lat']]
            addresses_list_df_scoped.rename(columns={'street': 'nom_voie'}, inplace=True)
            # Concatenate 2 parts
            complete_merged_df = pd.concat([perfect_match_merged_df, addresses_list_df_scoped], sort=True)
        complete_merged_df.drop_duplicates(subset=['lat', 'lon'], inplace=True)
        # Update run time count
        self.run_time_ratio += 5
        return complete_merged_df

    def apply_polling_stations_redistricting(self):
        """
        Method which apply a redistricting to addresses labels which have more than one polling station
        """
        street_dict = {}  # street: polling_station_id
        for ri, rd in self.addresses_list_df.iterrows():
            street_series = self.addresses_list_df[self.addresses_list_df.street == rd.street]
            # Filter duplicates streets labels
            duplicates_condition = any(k for k in street_dict.keys() if rd.street.upper() in k)
            # Redistricting if a street has more than 1 polling station and is not already in dictionary
            if len(street_series.polling_station_id.unique()) > 1 and duplicates_condition is False:
                # Update polling stations id by reassigning then for each even or odd streets labels
                for inner_ri, inner_rd in street_series.iterrows():
                    nb_start = inner_rd.number_start
                    nb_stop = inner_rd.number_stop
                    new_place_uid = str(inner_rd.place_uid) + str(inner_rd.polling_station_id).zfill(3)
                    if inner_rd.number_stop == 9999:
                        nb_stop = len(self.geocoded_df[self.geocoded_df.addresses.str.contains(inner_rd.street.upper())])
                    if inner_rd.even_numbers_only:
                        even_streets = [str(s) + ' ' + inner_rd.street.upper() for s in range(nb_start, nb_stop + 1, 2)]
                        even_street_dict = {even_street_label: new_place_uid for even_street_label in even_streets}
                        street_dict.update(even_street_dict)
                    if inner_rd.odd_numbers_only is True and inner_rd.even_numbers_only is False:
                        odd_streets = [str(s) + ' ' + inner_rd.street.upper() for s in range(nb_start, nb_stop + 1, 2)]
                        odd_street_dict = {odd_street_label: new_place_uid for odd_street_label in odd_streets}
                        street_dict.update(odd_street_dict)
        self.geocoded_df.place_uid = self.geocoded_df.addresses.map(street_dict).fillna(self.geocoded_df.place_uid)
        # Update run time count
        self.run_time_ratio += 5

    def build_geocoded_csv(self, export=False):
        """
        Method which build a geocoded dataframe
        :param export: boolean which enable export a geocoded dataframe as csv file (if it's False, then this method
        returns a geocoded dataframe)
        """
        complete_merged_df = self.complete_merge()
        complete_merged_df = complete_merged_df[~complete_merged_df['nom_voie'].isin(self.addresses_not_found)]
        addresses_list = []
        place_uid_list = []
        for row_idx, row_data in complete_merged_df.iterrows():
            place_uid_built = self.place_uid_pattern+str(row_data.polling_station_id).zfill(3)
            address_built = row_data.nom_voie.upper()
            if not np.isnan(row_data.numero):
                address_built = str(int(row_data.numero)) + ' ' + address_built
            addresses_list.append(address_built)
            place_uid_list.append(place_uid_built)
        complete_merged_df['addresses'] = addresses_list
        complete_merged_df['place_uid'] = place_uid_list

        self.geocoded_df = complete_merged_df[['addresses', 'place_uid', 'lat', 'lon']]
        # Apply redistricting to geocoded dataframe
        self.apply_polling_stations_redistricting()
        # Filter non-valid coordinates based on 2 conditions
        non_null_condition = ((self.geocoded_df.lat > 0) & (self.geocoded_df.lon > 0))
        consistency_condition = (self.geocoded_df.lat > self.geocoded_df.lon)
        self.geocoded_df = self.geocoded_df[non_null_condition & consistency_condition]
        self.ungeocoded_df = self.addresses_list_df[self.addresses_list_df.street.isin(self.addresses_not_found)]
        # Remove municipal code & city label columns
        self.ungeocoded_df = self.ungeocoded_df[self.ungeocoded_df.columns[:-2]]
        # Update run time count
        self.run_time_ratio += 5
        if export:
            self.geocoded_df.to_csv('{}_geocoded.csv'.format(self.location_label), index=False, sep=';')
        else:
            return self.geocoded_df

    def compute_geocoding_ratio(self):
        """
        Checking method which compute addresses geocoded ratio
        :return: a ratio as float
        """
        self.total_addresses_from_list = len(self.addresses_list_df.street.unique().tolist())
        self.total_ungeocoded_addresses_from_list = len(self.ungeocoded_df.street.unique().tolist())
        return round((1 - self.total_ungeocoded_addresses_from_list / self.total_addresses_from_list) * 100, 1)
