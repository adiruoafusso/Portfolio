import geopandas
from keplergl import KeplerGl
from geomodules.cartographer.utils.geometry_operations import reverse_layers_geodetic_system_conversion_from_geometry

########################################################################################################################
#                                            Data visualization                                                        #
########################################################################################################################


def map_geometry(main_layer, added_layers=None, main_label=None, second_label=None, map_height=800, reprojected=False):
    """
    Data visualization function which plot an interactive map within a jupyter notebook by using kepler.gl python
    module.

    Plotted maps_data can be exported as HTML files by using export_map_as_html function from geometry_files_writers module
    N.B : by default KeplerGl only read WGS84, if another CRS is used, then it must be reconverted to WGS84 CRS.

    :param main_layer: a main layer (a single geodataframe or a dictionary with a unique geodataframe)
    :param added_layers: an added layer (could be a dictionary of geodataframes or a single one)
           N.B : a nested dictionary has a layer label as key and a geodataframe as value
    :param main_label: layer's main label (str)
    :param second_label : second layer's label (str)
    :param map_height: map height (int)
    :param reprojected: boolean which identified geodetic system conversion (bool)

    :return: a map
    """
    # Main map static
    map_data = ""
    # Reverse geodetic system conversion (if the coordinate reference system has been reprojected).
    if reprojected:
        layers_dict = ""
        if type(added_layers) is dict:
            layers_dict = {'main_layer': main_layer, 'added_layers': added_layers}
        elif isinstance(added_layers, geopandas.geodataframe.GeoDataFrame):
            layers_dict = {'main_layer': main_layer, 'added_layers': {second_label: added_layers}}
        elif added_layers is None:
            layers_dict = {'main_layer': main_layer}
        else:
            assert "Must pass a (nested)dictionary with GeoDataFrames as values or a single GeoDataFrame !"
        # Recursive function which reverse geodetic system by reconvert layers coordinates (default CRS used is WGS84)
        reverse_layers_geodetic_system_conversion_from_geometry(layers_dict)

    # Give default labels to main & second layers
    if main_label is None:
        main_label = "Layer n°1"
    if second_label is None:
        second_label = "Layer n°2"

    # Build map by creating a main layer
    if type(main_layer) is dict:
        main_layer[list(main_layer.keys())[0]] = list(main_layer.values())[0].to_json()
        map_data = KeplerGl(height=map_height, data=main_layer)
    elif isinstance(main_layer, geopandas.geodataframe.GeoDataFrame):
        map_data = KeplerGl(height=map_height, data={main_label: main_layer.to_json()})
    else:
        assert "Must pass a dictionary with a GeoDataFrame as value or a single GeoDataFrame !"

    # Map added layers static (recursively if added layers is a nested or simple dictionary,
    # otherwise it just adds the second layer as a unique geodataframe)
    if type(added_layers) is dict:
        def add_map_data_recursively(add_layer):
            """
            Recursive function which map additional layers
            :param add_layer: an additional layer (dict: key: layer label value: geodataframe)
            """
            for layer_label, layer_data in add_layer.items():
                if type(layer_data) is dict:
                    add_map_data_recursively(layer_data)
                else:
                    # KeplerGL method which add a geodataframe as map layer
                    map_data.add_data(layer_data.to_json(), name=layer_label)
        add_map_data_recursively(added_layers)
    elif isinstance(added_layers, geopandas.geodataframe.GeoDataFrame):
        map_data.add_data(added_layers.to_json(), name=second_label)
    else:
        assert "Must pass a (nested)dictionary with GeoDataFrames as values or a single GeoDataFrame !"
    return map_data
