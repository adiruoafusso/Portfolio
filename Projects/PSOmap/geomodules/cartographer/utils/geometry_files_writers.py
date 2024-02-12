import os
import osmnx as ox
import pickle


########################################################################################################################
#                                               PICKLED FILES                                                          #
########################################################################################################################

def pickle_writer(nom_fichier, objet, pickle_data_path):
    """

    :param nom_fichier:
    :param objet:
    :param pickle_data_path:
    :return:
    """
    with open(os.path.join(pickle_data_path, nom_fichier), 'wb') as file:
        pickle.dump(objet, file, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_loader(file, pickle_data_path):
    """

    :param file:
    :param pickle_data_path:
    :return:
    """
    with open(os.path.join(pickle_data_path, file), 'rb') as file:
        object_unpickled = pickle.load(file)
    return object_unpickled

########################################################################################################################
#                                                SHAPEFILES                                                            #
########################################################################################################################


def convert_geodataframe_to_shapefile(gdf, filename, folder=None):
    """
    :param gdf:
    :param filename:
    :param folder:
    :return:
    """
    ox.save_gdf_shapefile(gdf, filename, folder)


def write_shapefiles_from_layers(layers_dict, folder=None):  # Transform as recursive function
    """

    :param layers_dict:
    :param folder:
    :return:
    """
    for layer_label, layer_data in layers_dict.items():
        if type(layer_data) is dict:
            write_shapefiles_from_layers(layer_data)
        else:
            convert_geodataframe_to_shapefile(layer_data, layer_label, folder)

########################################################################################################################
#                                                 HTML MAPS                                                            #
########################################################################################################################


def export_map_as_html(map_data, map_name, folder='/home/adil/Bureau/psomap/static/maps_data/'):
    """

    :param map_data:
    :param map_name:
    :param folder:
    :return:
    """
    return map_data.save_to_html(file_name=folder+'{}.html'.format(map_name))
