import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import shapely
from shapely.strtree import STRtree
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, shape, JOIN_STYLE
from shapely.ops import linemerge, unary_union, polygonize, nearest_points, cascaded_union
from shapely.wkb import loads
from pyproj import Proj, transform
from itertools import combinations

from geomodules.cartographer.config.config_file import geodetic_sys_refs

########################################################################################################################
#                                       COORDINATES TRANSFORMATIONS                                                    #
########################################################################################################################


def round_geometry_coordinates(geometry, precision):
    """

    :param geometry:
    :param precision:
    :return:
    """
    geometry = ox.round_shape_coords(geometry, precision)
    return geometry


def convert_geodetic_system_from_coordinates(latitude, longitude, from_epsg_ref='WGS84', to_epsg_ref='RGF93'):
    """

    :param latitude:
    :param longitude:
    :param from_epsg_ref:
    :param to_epsg_ref:
    :return:
    """
    origin_geodesic_ref = Proj(init='{}'.format(geodetic_sys_refs[from_epsg_ref]))
    destination_geodesic_ref = Proj(init='{}'.format(geodetic_sys_refs[to_epsg_ref]))
    return transform(origin_geodesic_ref, destination_geodesic_ref, latitude, longitude)


def convert_geodetic_system_from_geometry(gdf, epsg_ref='RGF93'):
    """
    :param gdf: geocoded static from pandas or geopandas dataframe
    :param epsg_ref: (default is RGF93)
    """
    gdf.geometry = gdf.geometry.to_crs({'init': geodetic_sys_refs[epsg_ref]})


def reverse_layers_geodetic_system_conversion_from_geometry(layers_dict, epsg_ref='WGS84'):
    """
    A recursive function which convert geometry geodetic system in a nested layers dictionary
    :param layers_dict:
    :param epsg_ref:
    :return:
    """
    for layer_data in layers_dict.values():
        if type(layer_data) is dict:
            reverse_layers_geodetic_system_conversion_from_geometry(layer_data, epsg_ref)
        else:
            convert_geodetic_system_from_geometry(layer_data, epsg_ref)


def add_crs_to_gdf(gdfs, geodetic_system_converted=False):
    """

    :param gdfs:
    :param geodetic_system_converted:
    :return:
    """
    if type(gdfs) is not list:
        gdfs = [gdfs]
    for gdf in gdfs:
        if geodetic_system_converted:
            gdf.crs = {'init': geodetic_sys_refs['RGF93']}
        else:
            gdf.crs = {'init': geodetic_sys_refs['WGS84']}

########################################################################################################################
#                                         GEOMETRY TRANSFORMATIONS                                                     #
########################################################################################################################


def add_id_poly_column_from_index(gdf, inplace=True):
    """

    :param gdf:
    :param inplace:
    :return:
    """
    gdf.reset_index(inplace=True)
    gdf.rename(columns={'index': 'id_poly'}, inplace=True)
    if inplace is False:
        return gdf


def drop_geometry_duplicates(gdf, geometry_column_label='geometry', inplace=True):
    """

    :param gdf:
    :param geometry_column_label:
    :param inplace:
    :return:
    """
    gdf['geometry_str'] = gdf[geometry_column_label].map(lambda p: p.wkt)
    gdf.drop_duplicates(subset='geometry_str', inplace=True)
    gdf.drop(columns='geometry_str', inplace=True)
    if inplace is False:
        return gdf


def split_polygon_by_lines(polygon_gdf, lines_gdf):
    """

    :param polygon_gdf:
    :param lines_gdf:
    :return:
    """
    lines_list = list(lines_gdf.geometry)
    lines_list.append(polygon_gdf.geometry[0].boundary)
    merged_lines = linemerge(lines_list)
    border_lines = unary_union(merged_lines)  # Cost +++ (5-6 min) USE NUMBA ???
    decomposition = polygonize(border_lines)
    shattered_polygons = [x for x in decomposition]
    boundaries_poly_split_by_roads_gdf = gpd.GeoDataFrame(geometry=shattered_polygons)
    return boundaries_poly_split_by_roads_gdf


def merge_polygons(polygons_gdf):  # Union operation
    """

    :param polygons_gdf:
    :return:
    """
    return gpd.GeoDataFrame(geometry=[cascaded_union(polygons_gdf.geometry)])


def merge_lines(lines_gdf):
    """

    :param lines_gdf:
    :return:
    """
    multi_line = MultiLineString(list(lines_gdf.geometry))
    merged_line = linemerge(multi_line)
    merged_lines_gdf = gpd.GeoDataFrame(geometry=[merged_line])
    return merged_lines_gdf


def get_geometric_difference(original_list, alternative_list=None):
    """

    :param original_list:
    :param alternative_list:
    :return:
    """
    diff_list = []
    for poly in original_list:
        if not any(p.equals(poly) for p in [diff_list if alternative_list is None else alternative_list][0]):
            diff_list.append(poly)
    return diff_list


def get_geometric_difference_from_binaries(original_list, alternative_set=None):
    """

    :param original_list:
    :param alternative_set:
    :return:
    """
    diff_set = set()
    for poly in original_list:
        if poly.wkb not in [diff_set if alternative_set is None else alternative_set][0]:
            diff_set.add(poly.wkb)
    return [loads(p) for p in diff_set]


def get_geometry_nearest_neighbors(base_gdf, candidates_gdf, distance_threshold, geodetic_system_converted=False):
    """

    :param base_gdf:
    :param candidates_gdf:
    :param distance_threshold:
    :param geodetic_system_converted:
    :return:
    """
    originals_polygons = base_gdf.geometry
    tree = STRtree(originals_polygons)
    # Get list of binaries polygons
    polygons_binaries = []
    place_uids = []
    for row_index, row_data in candidates_gdf.iterrows():
        candidate_geometry = row_data.geometry
        # Querying nearest base geometries from a candidate geometry
        nearest_geometries = tree.query(candidate_geometry)
        polygons_found = [p.wkb for p in nearest_geometries if p.distance(candidate_geometry) <= distance_threshold]
        polygons_binaries.extend(polygons_found)
        place_uids.extend([row_data.place_uid] * len(polygons_found))
    # Filter uniques matched polygons
    uniques_polygons_gdf = gpd.GeoDataFrame({'place_uid': place_uids, 'binary_geom': polygons_binaries})
    uniques_polygons_gdf.drop_duplicates(subset='binary_geom', inplace=True)
    uniques_polygons_binaries = uniques_polygons_gdf['binary_geom'].tolist()
    uniques_polygons_gdf['binary_geom'] = [loads(x) for x in uniques_polygons_gdf['binary_geom']]
    uniques_polygons_gdf.rename(columns={'binary_geom': 'geometry'}, inplace=True)
    uniques_polygons = uniques_polygons_gdf.shape[0]
    # Filter unmatched polygons
    diff_polygons = get_geometric_difference_from_binaries(originals_polygons, uniques_polygons_binaries)
    # Build geodataframes
    polygons_matched_gdf = uniques_polygons_gdf
    polygons_unmatched_gdf = gpd.GeoDataFrame(geometry=diff_polygons)
    # Define coordinates reference system for each geodataframes
    add_crs_to_gdf([polygons_matched_gdf, polygons_unmatched_gdf], geodetic_system_converted)
    # Checking lengths between original polygons dataframe and matched & unmatched polygons count
    computed_polygons_count = len(diff_polygons) + uniques_polygons
    total_polygons = base_gdf.shape[0]
    if computed_polygons_count == total_polygons:
        return {'matched': polygons_matched_gdf, 'unmatched': polygons_unmatched_gdf}
    else:
        print("Lengths inequals ! {} vs {}".format(computed_polygons_count, total_polygons))


def spatial_filter(covering_polygons, covered_polygons, geodetic_system_converted=False):
    """

    :param covering_polygons:
    :param covered_polygons:
    :param geodetic_system_converted:
    :return:
    """
    # Polygons list which are covered by one or many polygons (or not covered at all)
    covered_polygons_by_one = []
    covered_polygons_by_many = []
    uncovered_polygons = []
    # Place uid list for each covered polygons list
    place_uids_for_covered_polygons_by_one = []
    place_uids_for_covered_polygons_by_many = []
    # Use a R-tree in order to find overlapped polygons
    tree = STRtree(covering_polygons.geometry)
    # Main searching loop
    for recovered_poly in covered_polygons.geometry:
        # Use Well-known-binary format in order to perform set operation later on
        covering_polygons_found = [p.wkb for p in tree.query(recovered_poly) if p.contains(recovered_poly) is True]
        recovered_poly_binary = recovered_poly.wkb
        # Reload polygons from binaries in order to find associated place uid
        covering_polygons_found_reloaded = [loads(x) for x in covering_polygons_found]
        place_uids = []
        # Add this line into get_geometry_nearest_neighbors
        for polygon_reloaded in covering_polygons_found_reloaded:
            place_uids.extend([p for p in covering_polygons[covering_polygons.geometry == polygon_reloaded].place_uid])
        # Define specifics cases built in FME algorithm
        if len(covering_polygons_found) == 1:
            covered_polygons_by_one.append(recovered_poly_binary)
            place_uids_for_covered_polygons_by_one.extend(place_uids)
        elif len(covering_polygons_found) > 1:
            # np.repeat(recovered_poly_binary, len(place_uid)).tolist() is slower but maybe "safer"
            covered_polygons_by_many.extend([recovered_poly_binary]*len(place_uids))
            place_uids_for_covered_polygons_by_many.extend(place_uids)
        else:
            uncovered_polygons.append(recovered_poly_binary)
    # Create main geodataframe from polygons lists
    covered_polygons_by_one_gdf = gpd.GeoDataFrame({'place_uid': place_uids_for_covered_polygons_by_one,
                                                    'binary_geom': covered_polygons_by_one})
    covered_polygons_by_many_gdf = gpd.GeoDataFrame({'place_uid': place_uids_for_covered_polygons_by_many,
                                                     'binary_geom': covered_polygons_by_many})
    uncovered_polygons_gdf = gpd.GeoDataFrame(geometry=list([loads(p) for p in set(uncovered_polygons)]))
    # Drop duplicated geometry and reload polygons objects from binaries for each geodataframes
    for covered_polygons_gdf in [covered_polygons_by_one_gdf, covered_polygons_by_many_gdf]:
        covered_polygons_gdf.drop_duplicates(subset='binary_geom', inplace=True)
        covered_polygons_gdf['binary_geom'] = [loads(x) for x in covered_polygons_gdf['binary_geom']]
        covered_polygons_gdf.rename(columns={'binary_geom': 'geometry'}, inplace=True)
    # Build a dictionary which contains covered and uncovered polygons geodataframes
    spatial_gdfs_dict = {'one': covered_polygons_by_one_gdf,
                         'many': covered_polygons_by_many_gdf,
                         'uncovered': uncovered_polygons_gdf}
    # Add coordinate reference system for each geodataframes
    add_crs_to_gdf(list(spatial_gdfs_dict.values()), geodetic_system_converted)
    return spatial_gdfs_dict


def dissolve_polygons_by_place_uid(polygons_gdf):
    """

    :param polygons_gdf:
    :return:
    """
    # Dissolve by place_uid
    matched_polygons_dissolved_gdf = polygons_gdf.dissolve(by='place_uid')
    matched_polygons_dissolved_gdf.reset_index(inplace=True)
    dissolved_areas = []
    dissolved_areas_place_uids = []
    for row_index, row_data in matched_polygons_dissolved_gdf.iterrows():
        if isinstance(row_data.geometry, shapely.geometry.polygon.Polygon) is False:
            dissolved_areas.extend(list(row_data.geometry))
            place_uids = np.repeat(row_data.place_uid, len(list(row_data.geometry)))
            dissolved_areas_place_uids.extend(place_uids)
        else:
            dissolved_areas.append(row_data.geometry)
            dissolved_areas_place_uids.append(row_data.place_uid)
    # Build dissolved areas geodataframe
    dissolved_areas_gdf = gpd.GeoDataFrame({'place_uid': dissolved_areas_place_uids}, geometry=dissolved_areas)
    return dissolved_areas_gdf


def get_geometry_intersections(gdf, geometry_list=None, intersections_type='lines'):
    """

    :param gdf:
    :param geometry_list:
    :param intersections_type:
    :return:
    """
    if geometry_list is None:
        geometry_list = gdf.geometry.tolist()
    # Get intersections from dissolved areas
    intersected_geometries = []
    intersected_geometries_place_uids = []
    intersected_geometries_idx = []
    geometry_types = tuple()
    # Define geometry class types
    if intersections_type == 'lines':
        geometry_types = (shapely.geometry.linestring.LineString, shapely.geometry.multilinestring.MultiLineString)
    elif intersections_type == 'polygons':
        geometry_types = (shapely.geometry.polygon.Polygon, shapely.geometry.multipolygon.MultiPolygon)
    else:
        assert "{} geometry type is not currently implemented yet".format(intersections_type)
    # Build conditional function which filter a geometry class by a tuple
    def geometry_type_condition(x): return isinstance(x, geometry_types)
    # Filter geometry class type
    combs = [c for c in combinations(geometry_list, 2) if geometry_type_condition(c[0].intersection(c[1])) is True]
    # Iterate over combinations tuples list
    for a, b in combs:
        intersections_geom = a.intersection(b)
        # intersected_place_uid = ""
        # Place uid affectation (get polling station which is most represented)
        if len(a.exterior.coords) > len(b.exterior.coords):
            intersected_place_uid = gdf[gdf.geometry == a].place_uid.tolist()
            intersected_idx = gdf[gdf.geometry == a].id_poly.tolist()
        else:
            intersected_place_uid = gdf[gdf.geometry == b].place_uid.tolist()
            intersected_idx = gdf[gdf.geometry == b].id_poly.tolist()
        # MultiLinestring/MultiPolygon type
        if isinstance(intersections_geom, (geometry_types[1])):
            intersected_geometries.extend(list(intersections_geom))
            if len(intersected_place_uid) == 1 and len(intersected_idx) == 1:
                intersected_place_uid = np.repeat(intersected_place_uid[0], len(list(intersections_geom)))
                intersected_idx = np.repeat(intersected_idx[0], len(list(intersections_geom)))
            intersected_geometries_place_uids.extend(intersected_place_uid)
            intersected_geometries_idx.extend(intersected_idx)
        # Linestring/Polygon type
        elif isinstance(intersections_geom, (geometry_types[0])):
            intersected_geometries_place_uids.append(intersected_place_uid[0])
            intersected_geometries_idx.append(intersected_idx[0])
            intersected_geometries.append(intersections_geom)

    intersected_geometries_gdf = gpd.GeoDataFrame({'place_uid': intersected_geometries_place_uids},
                                                  geometry=intersected_geometries,
                                                  index=intersected_geometries_idx)
    return intersected_geometries_gdf


def common_limits_by_place_uid_tester(geometry_overlay_gdf):
    """

    :param geometry_overlay_gdf:
    :return:
    """
    # filter by uniques geometries
    # non-overlapped (areas with no common limits)
    valid_geometries_gdf = geometry_overlay_gdf[geometry_overlay_gdf.place_uid.isnull()]
    # overlapped (areas with common limits)
    non_valid_geometries_gdf = geometry_overlay_gdf[~geometry_overlay_gdf.place_uid.isnull()]
    common_limits_dict = {'passed': valid_geometries_gdf, 'failed': non_valid_geometries_gdf}  # no place_uid/ with ...
    return common_limits_dict


def rebuild_polygon_from_exterior_coordinates(polygon):
    """
    Function which fill holes in polygons interiors by creating a new polygon based on its previous exterior coordinates
    :param polygon: a polygon with holes
    :return: a polygon with holes filled
    """
    if len(list(polygon.interiors)) > 0:
        return Polygon(list(polygon.exterior.coords))
    else:
        return polygon


def remove_slivers(polygon, eps=0.001):
    """
    Function which remove slivers from polygon based on a specific tolerance
    :param polygon: a polygon which contains slivers
    :param eps: epsilon that is approx. the width of slivers, e.g. 1 mm
    :return: a cleaned polygon
    """
    return polygon.buffer(eps, 1, join_style=JOIN_STYLE.mitre).buffer(-eps, 1, join_style=JOIN_STYLE.mitre)


def fill_holes_within_polygons_interiors(polygons_gdf):
    """

    :param polygons_gdf:
    :return:
    """
    polygons_gdf.geometry = [rebuild_polygon_from_exterior_coordinates(p) for p in polygons_gdf.geometry]


def extract_gaps_from_polygons(polygons_gdf, layer_gdf):
    """

    :param polygons_gdf: main geodataframe which contains polygons with holes or/and slivers
    :param layer_gdf: second geodataframe which will be used as a differential layer
    :return:
    """
    extracted_gaps_gdf = gpd.overlay(polygons_gdf, layer_gdf, how='symmetric_difference')
    extracted_gaps_gdf = extracted_gaps_gdf.loc[:, ('place_uid', 'geometry')]
    # Extract polygons from multi polygons
    polygons_gaps = []
    place_uids = []
    for row_idx, row_data in extracted_gaps_gdf.iterrows():
        row_geom = row_data.geometry
        row_place_uid = row_data.place_uid
        if isinstance(row_geom, shapely.geometry.multipolygon.MultiPolygon):
            for polygon in list(row_geom):
                polygons_gaps.append(polygon)
                place_uids.append(row_place_uid)
        else:
            polygons_gaps.append(row_geom)
            place_uids.append(row_place_uid)

    extracted_gaps_as_polygons_gdf = gpd.GeoDataFrame({"place_uid": place_uids}, geometry=polygons_gaps)
    if extracted_gaps_as_polygons_gdf.place_uid.isna().all() is False:
        extracted_gaps_as_polygons_gdf.drop_duplicates(subset='place_uid', inplace=True)
    return extracted_gaps_as_polygons_gdf


def transform_polygons_as_lines(polygons_gdf):
    """

    :param polygons_gdf:
    :return:
    """
    place_uids = []
    lines = []
    for row_idx, row_data in polygons_gdf.iterrows():
        polygon_bounds = row_data.geometry.boundary
        if polygon_bounds.type == 'MultiLineString':
            for line in polygon_bounds:
                lines.append(line)
                place_uids.append(row_data.place_uid)
        else:
            lines.append(polygon_bounds)
            place_uids.append(row_data.place_uid)
    lines_gdf = gpd.GeoDataFrame({'place_uid': place_uids}, geometry=lines, crs=polygons_gdf.crs)
    return lines_gdf


########################################################################################################################
#                                            DATA ANALYSIS                                                             #
########################################################################################################################

def nearest_neighbor_search(origin, destinations):
    """
    :param origin:
    :param destinations:
    :return:
    """
    nearest_geometry = [nearest_points(orig, dest) for orig, dest in zip(origin.geometry, destinations.geometry)]
    points_list = [x[1] for x in nearest_geometry]
    nearest_neighbor_gdf = gpd.GeoDataFrame(geometry=points_list, crs=origin.crs)
    return nearest_neighbor_gdf
