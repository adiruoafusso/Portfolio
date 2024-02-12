import geopandas as gpd
import hdbscan
import osmnx as ox
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, shape
from utils.psomap_utilities import get_coordinates_columns_labels
from geomodules.cartographer.config.config_file import *
from geomodules.cartographer.utils.geometry_files_writers import *
from geomodules.cartographer.utils.geometry_operations import *
from geomodules.cartographer.utils.geometry_visualizer import map_geometry


class PSOCartographer:
    """
    """

    def __init__(self, geocoded_df, location, country='France', geodetic_system_converted=True, filter_outliers=True):
        """
        :param geocoded_df: a geocoded dataframe (with FrenchGeocoder or GermanGeocoder class)
        :param location: locality's label
        :param country: country's label (France or Germany)
        :param geodetic_system_converted: boolean which enable/disable geodetic system conversion
        """
        self.geocoded_df = geocoded_df
        self.polling_stations = self.geocoded_df.place_uid.unique().tolist()
        self.polling_stations_count = len(self.polling_stations)
        # TO DO : add word translator (bind location & country as unique str)
        self.location = location
        self.country = country
        self.geodetic_system_converted = geodetic_system_converted
        self.longitude_col, self.latitude_col = gpd_coords_cols_labels
        # Coordinates columns detection & normalisation (renaming)
        self.normalize_coordinates_columns_labels()
        if filter_outliers:
            self.geocoded_df = self.filter_outliers_with_hdbscan()
        # Main input static
        self.location_roads_gdf_dict = ""
        self.location_boundaries_gdf = ""
        self.addresses_points_gdf = ""
        self.addresses_polygons_gdf = ""
        # P1 FME dynamics inputs
        self.unmatched_polygons_gdf = ""
        self.matched_polygons_gdf = ""
        self.main_input_geometries_layers = ""
        # P2
        self.spatial_dict = ""
        self.assigned_polygons = ""
        # P3 latest optimisations
        self.polygons_aggregated_by_place_uid_dict = ""
        self.reassigned_polygons_by_total_points_dict = ""

    ####################################################################################################################
    #                                            Data extractors                                                       #
    ####################################################################################################################

    # TO DO : REFACTOR
    def get_boundaries_from_location(self, file_type='gdf'):
        """
        Method which extract location boundaries as a unique polygon. Could be saved as a shapefile or geodataframe.
        If file_type is gdf, then return a geodataframe.
        :param file_type: file type export (shp: shapefile, gdf: geodataframe)
        :return: a geodataframe which contains location boundaries
        """
        # Use osmnx function in order to create a geodataframe from location & country label which contains location
        # boundaries as polygon. (cf : https://github.com/gboeing/osmnx/blob/master/osmnx/core.py)
        self.location_boundaries_gdf = ox.gdf_from_place('{}, {}'.format(self.location, self.country))
        # Manage saving type
        if file_type == 'shp':
            # Create a specific folder based on location label and boundaries tag
            boundaries_folder_name = self.location + '_boundaries'
            ox.save_gdf_shapefile(self.location_boundaries_gdf, boundaries_folder_name, folder=None)
        elif file_type == 'gdf':
            return self.location_boundaries_gdf

    # TO DO : REFACTOR
    def get_roads_from_location(self, file_type='gdf', network_type='all_private', geodf_type='edges'):
        """
        Method which extract all ways & roads for a specific location, as lines or points.
        Could be saved as a shapefile or geodataframe. If file_type is gdf, then return a geodataframe.
        :param file_type: file type export (shp: shapefile, gdf: geodataframe)
        :param network_type: type of street network (cf : https://github.com/gboeing/osmnx/blob/master/osmnx/core.py)
        :param geodf_type: geodataframe type (will be used as key for roads dictionary)
        :return: a geodataframe which contains location's roads & ways
        """
        # Use osmnx function in order to create a graph from location & country label which contains
        # location's roads/ways.
        roads_graph = ox.graph_from_place('{}, {}'.format(self.location, self.country), network_type, retain_all=False)
        # Manage saving type
        if file_type == 'shp':
            # Create a specific folder based on location label and roads tag
            roads_folder_name = self.location + '_roads'
            ox.save_graph_shapefile(roads_graph, roads_folder_name, folder=None)
        elif file_type == 'gdf':
            graph_data_types = ['nodes', 'edges']
            # Convert a network graph as a list of geodataframes (split as nodes & edges geodataframes)
            geodf_types = ox.graph_to_gdfs(roads_graph)
            # Build a dictionary which store geodataframes roads/ways as nodes (points) and edges (lines).
            self.location_roads_gdf_dict = {g_type: gdf_type for g_type, gdf_type in zip(graph_data_types, geodf_types)}
            return self.location_roads_gdf_dict[geodf_type]

    ####################################################################################################################
    #                                            Data transformations                                                  #
    ####################################################################################################################

    def normalize_coordinates_columns_labels(self):
        """
        Method which rename coordinates columns labels by using geopandas default nomenclature :
        - x : longitude column
        - y : latitude column
        """
        x_col, y_col = self.longitude_col, self.latitude_col
        coords_cols_labels = get_coordinates_columns_labels(self.geocoded_df)
        # If first col value is lower than the second one, then it's a longitude. Otherwise it's a latitude
        if self.geocoded_df[coords_cols_labels[0]].iloc[0] > self.geocoded_df[coords_cols_labels[1]].iloc[0]:
            self.geocoded_df.rename(columns={coords_cols_labels[0]: y_col, coords_cols_labels[1]: x_col}, inplace=True)
        else:
            self.geocoded_df.rename(columns={coords_cols_labels[0]: x_col, coords_cols_labels[1]: y_col}, inplace=True)

    def filter_outliers_with_hdbscan(self, quantile_value=0.75):  # old value 0.9
        """
        Method which use HDBSCAN clustering algorithm in order to filter outliers
        :param quantile_value: quantile value which filter outliers (default is D9)
        Documentation :
        - https://github.com/scikit-learn-contrib/hdbscan
        - https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
        :return:
        """
        coordinates = self.geocoded_df[['x', 'y']].values
        min_points = min([self.geocoded_df[self.geocoded_df.place_uid == p].shape[0] for p in self.polling_stations])
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_points).fit(coordinates)
        threshold = pd.Series(hdbscan_clusterer.outlier_scores_).quantile(quantile_value)
        # Get outliers indexes by filtering values which are greater than quantile value
        outliers = np.where(hdbscan_clusterer.outlier_scores_ > threshold)[0]
        # Filter
        reduced_geocoded_df = self.geocoded_df[~self.geocoded_df.index.isin(outliers)]
        return reduced_geocoded_df

    def transform_geocoded_df_to_geopandas_df(self):
        """

        :return:
        """
        coordinates_as_points = [Point((x, y)) for x, y in zip(self.geocoded_df.x, self.geocoded_df.y)]
        transformed_gdf = gpd.GeoDataFrame({'place_uid': self.geocoded_df.place_uid,
                                            'geometry': coordinates_as_points})
        transformed_gdf.crs = {'init': geodetic_sys_refs['WGS84']}
        return transformed_gdf

    def transform_coordinates_to_geometry_for_each_place_uid(self, geometry_type='polygon', merge_place_uid=False):
        """
        :param geometry_type:
        :param merge_place_uid:
        :return: geodataframe or a dictionary which contains a geodataframe for each polling station id
        """
        coords_by_place_uid = {}
        for uid in list(self.geocoded_df.place_uid.unique()):
            x_wgs_per_uid = self.geocoded_df[self.longitude_col][self.geocoded_df.place_uid == uid]
            y_wgs_per_uid = self.geocoded_df[self.latitude_col][self.geocoded_df.place_uid == uid]
            geodf_addrs = gpd.GeoDataFrame(crs={'init': 'epsg:4326'})
            if geometry_type == 'polygon':
                polygons = [Polygon([(x, y) for x, y in zip(x_wgs_per_uid, y_wgs_per_uid)]).convex_hull]
                geodf_addrs['geometry'] = polygons
                coords_by_place_uid[uid] = geodf_addrs
            elif geometry_type == 'line':
                lines = [LineString([(x, y) for x, y in zip(x_wgs_per_uid, y_wgs_per_uid)])]
                geodf_addrs['geometry'] = lines
                coords_by_place_uid[uid] = geodf_addrs
            elif geometry_type == 'points':
                points = [Point(x, y) for x, y in zip(x_wgs_per_uid, y_wgs_per_uid)]
                geodf_addrs['geometry'] = points
                coords_by_place_uid[uid] = geodf_addrs
        if merge_place_uid:
            geodf_addrs = gpd.GeoDataFrame(geometry=[x.geometry.iloc[0] for x in list(coords_by_place_uid.values())])
            geodf_addrs['place_uid'] = self.geocoded_df.place_uid.unique().tolist()
            geodf_addrs.crs = {'init': geodetic_sys_refs['WGS84']}
            return geodf_addrs
        else:
            return coords_by_place_uid

    ####################################################################################################################
    #                                        BLOCK 1 BUILD INPUT DATA                                                  #
    ####################################################################################################################

    def build_main_input_data(self, rounded_geometry_coordinates=True, roads_type='all_private'):
        """

        :param rounded_geometry_coordinates:
        :param roads_type:
        :return:
        """
        # Location outlines representation (polygon)
        self.location_boundaries_gdf = self.get_boundaries_from_location(file_type='gdf')
        # Location roads representation (lines)
        roads_edges_gdf = self.get_roads_from_location(file_type='gdf', network_type=roads_type, geodf_type='edges')
        # roads_points_gdf = self.location_roads_gdf_dict['nodes']
        # Polling stations representation
        self.addresses_points_gdf = self.transform_geocoded_df_to_geopandas_df()
        self.addresses_polygons_gdf = self.transform_coordinates_to_geometry_for_each_place_uid(merge_place_uid=True)
        # Optional : convert coordinates geodesic system
        gdfs = [self.location_boundaries_gdf, roads_edges_gdf, self.addresses_points_gdf, self.addresses_polygons_gdf]
        if self.geodetic_system_converted:
            for gdf in gdfs:
                convert_geodetic_system_from_geometry(gdf)
                if rounded_geometry_coordinates:
                    try:
                        gdf.geometry = gdf.geometry.apply(lambda x: round_geometry_coordinates(x, 1))
                    except AttributeError as ae:
                        print(ae)
            # Update location roads dictionary if geodetic system is converted
            self.location_roads_gdf_dict['edges'] = roads_edges_gdf
        # Location outlines split by roads lines
        boundaries_polygons_split_by_roads_gdf = split_polygon_by_lines(self.location_boundaries_gdf, roads_edges_gdf)
        # Matched polygons with addresses points representation
        nearest_neighbors_polygons_dict = get_geometry_nearest_neighbors(boundaries_polygons_split_by_roads_gdf,
                                                                         self.addresses_points_gdf,
                                                                         10,
                                                                         self.geodetic_system_converted)
        self.unmatched_polygons_gdf = nearest_neighbors_polygons_dict['unmatched']
        self.matched_polygons_gdf = nearest_neighbors_polygons_dict['matched']
        # Build prototyping layers dict
        self.main_input_geometries_layers = {'main_layer': self.addresses_polygons_gdf,
                                             'added_layers': {'addresses as points': self.addresses_points_gdf,
                                                              'matched polygons': self.matched_polygons_gdf,
                                                              'unmatched polygons': self.unmatched_polygons_gdf}}

    # Bloc : Cas des polygones contenant des points (used in order to complete unmatched polygons list)
    def rebuild_unmatched_polygons_gdf(self, points_gdf=None, polygons_matched_gdf=None):
        """

        :param points_gdf:
        :param polygons_matched_gdf:
        :return:
        """
        if points_gdf is None and polygons_matched_gdf is None:
            points_gdf = self.addresses_points_gdf
            polygons_matched_gdf = self.matched_polygons_gdf

        points_gdf = points_gdf[['place_uid', 'geometry']]
        points_in_polys = gpd.sjoin(points_gdf, polygons_matched_gdf, op='within')
        # Aggregate by polygons
        points_in_polys = points_in_polys.dissolve(by='index_right')
        get_coords = lambda m: len(m.geoms) if isinstance(m, shapely.geometry.point.Point) is False else len(m.coords)
        points_in_polys['total_points'] = points_in_polys.geometry.apply(get_coords)
        # Sort points count per polygons
        points_in_polys.sort_values(by=['total_points'], inplace=True)
        points_in_polys = points_in_polys.reset_index()
        polygons_matched_gdf_reindex = polygons_matched_gdf.reset_index().rename(columns={'index': 'index_right'})
        # Merge again
        gdf_merged = points_in_polys.merge(polygons_matched_gdf_reindex, on=['index_right'])
        # select the polling station id that contains the most points
        polygons_assigned_raw = gdf_merged.drop_duplicates(subset='index_right', keep="last")
        # save selected polygons as a subset which contains place_uid & polygons geometry
        cols_scoped = ['place_uid', 'geometry_y']
        polygons_assigned = polygons_assigned_raw[cols_scoped].rename(columns={'geometry_y': 'geometry'})
        polygons_assigned_gdf = gpd.GeoDataFrame({'place_uid': polygons_assigned.place_uid,
                                                  'geometry': polygons_assigned.geometry})
        # Get recalcitrant unmatched polygons which will be added to the main unmatched list
        polygons_dest_list = polygons_assigned.geometry.tolist()
        polygons_orig_list = polygons_matched_gdf.geometry.tolist()
        recalcitrant_unmatched = get_geometric_difference(polygons_orig_list, polygons_dest_list)
        # Update total unmatched polygons based on overwriting the original unmatched_polygons_gdf
        unmatched_polygons_list = self.unmatched_polygons_gdf.geometry.tolist()
        self.unmatched_polygons_gdf = gpd.GeoDataFrame(geometry=unmatched_polygons_list + recalcitrant_unmatched)
        # Define coordinates representation system for each geodataframes
        add_crs_to_gdf([polygons_assigned_gdf, self.unmatched_polygons_gdf], self.geodetic_system_converted)
        self.assigned_polygons = polygons_assigned_gdf

    # TestFilter
    def find_overlapped_polygons(self):
        """

        :return:
        """
        self.spatial_dict = spatial_filter(self.addresses_polygons_gdf,
                                           self.unmatched_polygons_gdf,
                                           self.geodetic_system_converted)

    ####################################################################################################################
    #                                   BLOCK 2 : REBUILD POLYGONS SPLIT BY MANY PLACE UID                             #
    ####################################################################################################################

    def find_polygons_split_by_many_place_uid(self):
        """

        :return:
        """
        # Rebuild unmatched polygons list
        self.rebuild_unmatched_polygons_gdf()
        self.find_overlapped_polygons()
        covered_polygons_by_many_place_uid_gdf = self.spatial_dict['many']
        gdf_concatenated = pd.concat([covered_polygons_by_many_place_uid_gdf, self.addresses_polygons_gdf], sort=False)
        drop_geometry_duplicates(gdf_concatenated)
        add_id_poly_column_from_index(gdf_concatenated)
        polygons_list_extended = covered_polygons_by_many_place_uid_gdf.geometry.tolist() \
                                 + self.addresses_polygons_gdf.geometry.tolist()
        intersected_polygons_gdf = get_geometry_intersections(gdf_concatenated, polygons_list_extended, 'polygons')
        # Generate polygons indexes
        for gdf in [intersected_polygons_gdf, covered_polygons_by_many_place_uid_gdf]:
            gdf.pipe(add_id_poly_column_from_index)
        df_merged = pd.merge(covered_polygons_by_many_place_uid_gdf, intersected_polygons_gdf,
                             on=['id_poly', 'place_uid'], how='left')
        drop_geometry_duplicates(df_merged, geometry_column_label='geometry_x')
        df_merged = df_merged[['id_poly', 'place_uid', 'geometry_x']]
        df_merged.rename(columns={'geometry_x': 'geometry'}, inplace=True)
        df_merged['polygon_area'] = df_merged.geometry.map(lambda p: p.area)
        df_merged.sort_values(by='polygon_area', ascending=False, inplace=True)
        # normalize dataframe columns in order to concat them later
        df_merged = df_merged.loc[:, ('place_uid', 'geometry')]
        return df_merged

    ####################################################################################################################
    #                                      BLOCK 3 : FIND LARGEST COMMON LIMIT                                         #
    ####################################################################################################################

    def find_largest_common_limit(self):
        """

        :return:
        """
        uncovered_polygons_gdf = self.spatial_dict['uncovered']

        dissolved_areas_gdf = dissolve_polygons_by_place_uid(self.matched_polygons_gdf)

        for gdf in [uncovered_polygons_gdf, dissolved_areas_gdf]:
            gdf.pipe(add_id_poly_column_from_index)

        intersected_lines_gdf = get_geometry_intersections(dissolved_areas_gdf)
        # Add crs to new geodataframes
        gdf_list_with_no_crs = [dissolved_areas_gdf, intersected_lines_gdf]
        add_crs_to_gdf(gdf_list_with_no_crs, self.geodetic_system_converted)
        # Length calculator (filter linestrings)
        intersected_lines_gdf['line_length'] = intersected_lines_gdf.geometry.map(lambda line: line.length)
        # Get areas from lines overlayed
        areas_overlay_gdf = gpd.sjoin(intersected_lines_gdf, uncovered_polygons_gdf, how='right', op='intersects')
        drop_geometry_duplicates(areas_overlay_gdf)
        # Find common limits
        common_limits_dict = common_limits_by_place_uid_tester(areas_overlay_gdf)
        common_limits_dict['failed'] = common_limits_dict['failed'].sort_values(by='line_length', ascending=False)
        return common_limits_dict

    def nearest_polygons_treated_as_lines(self):
        """

        :return:
        """
        covered_polygons_by_many_gdf = self.find_polygons_split_by_many_place_uid()
        common_limits_dict = self.find_largest_common_limit()
        # normalize dataframe columns in order to concat them
        commons_limits_failed_gdf = common_limits_dict['failed'].loc[:, ('place_uid', 'geometry')]
        polygons_base = common_limits_dict['passed']
        polygons_candidates = [covered_polygons_by_many_gdf, commons_limits_failed_gdf, self.matched_polygons_gdf]
        polygons_candidates = pd.concat(polygons_candidates, sort=False)

        nearest_neighbors_polygons_dict = get_geometry_nearest_neighbors(polygons_base,
                                                                         polygons_candidates,
                                                                         100000,
                                                                         self.geodetic_system_converted)

        polygons_to_dissolve_1 = nearest_neighbors_polygons_dict['matched']
        polygons_to_dissolve_2 = commons_limits_failed_gdf
        polygons_to_dissolve_3 = self.spatial_dict['one']
        polygons_to_dissolve_4 = covered_polygons_by_many_gdf
        polygons_to_dissolve_5 = self.matched_polygons_gdf
        polygons_to_dissolve_chunks = [polygons_to_dissolve_1, polygons_to_dissolve_2, polygons_to_dissolve_3,
                                       polygons_to_dissolve_4, polygons_to_dissolve_5]
        polygons_to_dissolve = pd.concat(polygons_to_dissolve_chunks, sort=False)
        # print(polygons_base.shape, polygons_to_dissolve_1.shape) (4592, 5) (4586, 2)
        return polygons_to_dissolve

    ####################################################################################################################
    #                                   BLOCK 4 : DISSOLVE POLYGONS BY PLACE UID                                       #
    ####################################################################################################################

    def dissolve_nearest_neighbors_polygons_by_place_uid(self):
        """

        :return:
        """
        polygons_to_dissolve = self.nearest_polygons_treated_as_lines()
        dissolved_areas_gdf = dissolve_polygons_by_place_uid(polygons_to_dissolve)
        add_crs_to_gdf(dissolved_areas_gdf, self.geodetic_system_converted)
        return dissolved_areas_gdf

    def find_aggregated_place_uid(self):
        """

        :return:
        """
        dissolved_areas_gdf = self.dissolve_nearest_neighbors_polygons_by_place_uid()
        # Build a total column which store place uid count for each unique place uid
        dissolved_areas_gdf["total"] = dissolved_areas_gdf.groupby(['place_uid']).transform("count")
        # Dissolve geometry by place uid (polygons aggregation)
        dissolved_areas_by_place_uid = dissolved_areas_gdf.dissolve(by='place_uid')
        dissolved_areas_by_place_uid.reset_index(inplace=True)
        # Filter dissolved areas by place uid count
        dissolved_areas_by_many = dissolved_areas_by_place_uid[dissolved_areas_by_place_uid.total > 1]
        dissolved_areas_by_one = dissolved_areas_by_place_uid[dissolved_areas_by_place_uid.total == 1]
        self.polygons_aggregated_by_place_uid_dict = {'one': dissolved_areas_by_one, 'many': dissolved_areas_by_many}

    # Modification 24012017
    def filter_polygons_candidates_by_maximum_points(self):
        """

        :return:
        """
        self.find_aggregated_place_uid()
        # Get geodataframe where place uid had more than 1 polygon
        uid_many_poly = self.polygons_aggregated_by_place_uid_dict['many']
        # Break up polygons tuples for each place uid
        uid_many_poly_exploded = gpd.GeoDataFrame(uid_many_poly.geometry.explode())
        # Rename multi-index
        uid_many_poly_exploded.index.rename(['place_uid', 'candidates'], inplace=True)
        # Add a total column which store points count for each polygons
        uid_many_poly_exploded['total_points'] = uid_many_poly_exploded.geometry.map(lambda p: len(p.exterior.coords))
        # For each place uid, select a polygon candidate which had the highest points count
        max_candidates = uid_many_poly_exploded.loc[uid_many_poly_exploded.groupby(level=0)['total_points'].idxmax()]
        # Get others values
        others_candidates = uid_many_poly_exploded[~uid_many_poly_exploded.geometry.map(lambda p: max_candidates.geometry.contains(p).any())]
        # Build a translation dictionary which contains index row associated to a specific place_uid label
        place_uid_dict = {row_idx: row_data.place_uid for row_idx, row_data in uid_many_poly.iterrows()}
        for gdf in [max_candidates, others_candidates]:
            gdf.reset_index(inplace=True)
            gdf.loc[:, 'place_uid'] = gdf['place_uid'].map(place_uid_dict)
            gdf = gdf.loc[:, ('place_uid', 'geometry')]
        # Add crs to new geodataframes
        add_crs_to_gdf([max_candidates, others_candidates], self.geodetic_system_converted)
        self.reassigned_polygons_by_total_points_dict = {'uniques_candidates_based_on_max_points': max_candidates,
                                                         'other_candidates': others_candidates}

    def fill_gaps_between_polygons_areas(self):
        """

        :return:
        """
        self.filter_polygons_candidates_by_maximum_points()
        uid_one_poly = self.polygons_aggregated_by_place_uid_dict['one'].loc[:, ('place_uid', 'geometry')]
        max_candidates = self.reassigned_polygons_by_total_points_dict['uniques_candidates_based_on_max_points']
        areas_with_holes_gdf = pd.concat([uid_one_poly, max_candidates], sort=False)
        # Fill holes which are located in polygons interiors
        fill_holes_within_polygons_interiors(areas_with_holes_gdf)
        add_crs_to_gdf(areas_with_holes_gdf, self.geodetic_system_converted)
        areas_with_holes_gdf = areas_with_holes_gdf.loc[:, ('place_uid', 'geometry')]
        areas_with_holes_dissolved_gdf = areas_with_holes_gdf.dissolve(by='place_uid')
        areas_with_holes_dissolved_gdf.reset_index(inplace=True)
        return areas_with_holes_dissolved_gdf

    def reassign_unjoined_polygons_by_nearest_neighbors_largest_common_limit(self):
        """

        :return:
        """
        areas_with_holes_dissolved_gdf = self.fill_gaps_between_polygons_areas()
        others_candidates = self.reassigned_polygons_by_total_points_dict['other_candidates']
        concatenated_input = pd.concat([areas_with_holes_dissolved_gdf, others_candidates], sort=False)
        concatenated_input = concatenated_input.loc[:, ('place_uid', 'geometry')]
        # Remove gaps within polygons
        fill_holes_within_polygons_interiors(others_candidates)
        intersected_lines_gdf = transform_polygons_as_lines(concatenated_input)
        intersected_lines_gdf['line_length'] = intersected_lines_gdf.geometry.map(lambda line: line.length)
        # overlay
        areas_overlay_gdf = gpd.sjoin(others_candidates, intersected_lines_gdf, how='left', op='intersects')
        drop_geometry_duplicates(areas_overlay_gdf)
        areas_overlay_gdf = areas_overlay_gdf.sort_values(by='line_length', ascending=False)
        areas_overlay_gdf = areas_overlay_gdf[['place_uid_left', 'geometry']]
        areas_overlay_gdf.rename(columns={'place_uid_left': 'place_uid'}, inplace=True)
        return [areas_overlay_gdf, areas_with_holes_dissolved_gdf]

    def build_polling_stations_outlines(self, ways_type='drive'):
        """

        :return:
        """
        self.build_main_input_data(roads_type=ways_type)
        gdf_list = self.reassign_unjoined_polygons_by_nearest_neighbors_largest_common_limit()
        areas_overlay_gdf, areas_with_holes_dissolved_gdf = gdf_list
        areas_input_concatenated = pd.concat([areas_with_holes_dissolved_gdf, areas_overlay_gdf], sort=False)
        #
        fill_holes_within_polygons_interiors(areas_input_concatenated)
        areas_input_concatenated.geometry = areas_input_concatenated.geometry.map(lambda x: round_geometry_coordinates(x, 2))
        areas_input_concatenated = areas_input_concatenated[['place_uid', 'geometry']]
        areas_dissolved = areas_input_concatenated.dissolve(by='place_uid', aggfunc='sum')
        areas_dissolved.reset_index(inplace=True)
        areas_dissolved.geometry = areas_dissolved.geometry.map(lambda p: remove_slivers(p))
        map_name = '{}_polling_stations_outlines'.format(self.location)
        map_parameters = {'main_layer': {'addresses by polling stations': self.addresses_points_gdf},
                          'added_layers': {'polling stations outlines': areas_dissolved},
                          'reprojected': self.geodetic_system_converted}
        # Export a shapefiles
        location_folder = '/home/adil/Bureau/psomap/static/maps_data/{}/'.format(self.location)
        convert_geodataframe_to_shapefile(self.addresses_points_gdf,
                                          '{}_addresses_as_points'.format(self.location),
                                          folder=location_folder[:-1])
        convert_geodataframe_to_shapefile(areas_dissolved,
                                          '{}_polling_stations_outlines'.format(self.location),
                                          folder=location_folder[:-1])
        # Export map as HTML file
        export_map_as_html(map_geometry(**map_parameters), map_name, location_folder)
