
########################################################################################################################
#                                              Data structures                                                         #
########################################################################################################################

# Geopandas coordinates default columns labels
gpd_coords_cols_labels = ['x', 'y']

# Geodesic system references
geodetic_sys_refs = {'RGF93': 'epsg:2154',
                     'WGS84': 'epsg:4326'}
# Map configuration file
map_config = {'version': 'v1',
              'config': {'visState': {'filters': [],
                                      'layers': [{'id': '3bku88q',
                                                  'type': 'geojson',
                                                  'config': {'dataId': 'addresses by polling stations',
                                                             'label': 'addresses by polling stations',
                                                             'color': [231, 159, 213],
                                                             'columns': {'geojson': '_geojson'},
                                                             'isVisible': False,
                                                             'visConfig': {'opacity': 0.8,
                                                                           'thickness': 0.5,
                                                                           'strokeColor': None,
                                                                           'colorRange': {
                                                                               'name': 'Uber Viz Qualitative 4',
                                                                               'type': 'qualitative',
                                                                               'category': 'Uber',
                                                                               'colors': ['#12939A',
                                                                                          '#DDB27C',
                                                                                          '#88572C',
                                                                                          '#FF991F',
                                                                                          '#F15C17',
                                                                                          '#223F9A',
                                                                                          '#DA70BF',
                                                                                          '#125C77',
                                                                                          '#4DC19C',
                                                                                          '#776E57',
                                                                                          '#17B8BE',
                                                                                          '#F6D18A',
                                                                                          '#B7885E',
                                                                                          '#FFCB99',
                                                                                          '#F89570',
                                                                                          '#829AE3',
                                                                                          '#E79FD5',
                                                                                          '#1E96BE',
                                                                                          '#89DAC1',
                                                                                          '#B3AD9E'],
                                                                               'reversed': False},
                                                                           'strokeColorRange': {
                                                                               'name': 'Global Warming',
                                                                               'type': 'sequential',
                                                                               'category': 'Uber',
                                                                               'colors': ['#5A1846',
                                                                                          '#900C3F',
                                                                                          '#C70039',
                                                                                          '#E3611C',
                                                                                          '#F1920E',
                                                                                          '#FFC300']},
                                                                           'radius': 10,
                                                                           'sizeRange': [0, 10],
                                                                           'radiusRange': [0, 50],
                                                                           'heightRange': [0, 500],
                                                                           'elevationScale': 5,
                                                                           'stroked': False,
                                                                           'filled': True,
                                                                           'enable3d': False,
                                                                           'wireframe': False},
                                                             'textLabel': [{'field': None,
                                                                            'color': [255, 255, 255],
                                                                            'size': 18,
                                                                            'offset': [0, 0],
                                                                            'anchor': 'start',
                                                                            'alignment': 'center'}]},
                                                  'visualChannels': {
                                                      'colorField': {'name': 'place_uid', 'type': 'string'},
                                                      'colorScale': 'ordinal',
                                                      'sizeField': None,
                                                      'sizeScale': 'linear',
                                                      'strokeColorField': None,
                                                      'strokeColorScale': 'quantile',
                                                      'heightField': None,
                                                      'heightScale': 'linear',
                                                      'radiusField': None,
                                                      'radiusScale': 'linear'}},
                                                 {'id': 'pihjklc',
                                                  'type': 'geojson',
                                                  'config': {'dataId': 'polling stations outlines',
                                                             'label': 'polling stations outlines',
                                                             'color': [46, 42, 34],
                                                             'columns': {'geojson': '_geojson'},
                                                             'isVisible': True,
                                                             'visConfig': {'opacity': 0.8,
                                                                           'thickness': 0.5,
                                                                           'strokeColor': [19, 164, 171],
                                                                           'colorRange': {'name': 'Global Warming',
                                                                                          'type': 'sequential',
                                                                                          'category': 'Uber',
                                                                                          'colors': ['#5A1846',
                                                                                                     '#900C3F',
                                                                                                     '#C70039',
                                                                                                     '#E3611C',
                                                                                                     '#F1920E',
                                                                                                     '#FFC300']},
                                                                           'strokeColorRange': {
                                                                               'name': 'Global Warming',
                                                                               'type': 'sequential',
                                                                               'category': 'Uber',
                                                                               'colors': ['#5A1846',
                                                                                          '#900C3F',
                                                                                          '#C70039',
                                                                                          '#E3611C',
                                                                                          '#F1920E',
                                                                                          '#FFC300']},
                                                                           'radius': 10,
                                                                           'sizeRange': [0, 10],
                                                                           'radiusRange': [0, 50],
                                                                           'heightRange': [0, 500],
                                                                           'elevationScale': 5,
                                                                           'stroked': True,
                                                                           'filled': True,
                                                                           'enable3d': False,
                                                                           'wireframe': False},
                                                             'textLabel': [{'field': None,
                                                                            'color': [255, 255, 255],
                                                                            'size': 18,
                                                                            'offset': [0, 0],
                                                                            'anchor': 'start',
                                                                            'alignment': 'center'}]},
                                                  'visualChannels': {'colorField': None,
                                                                     'colorScale': 'quantile',
                                                                     'sizeField': None,
                                                                     'sizeScale': 'linear',
                                                                     'strokeColorField': None,
                                                                     'strokeColorScale': 'quantile',
                                                                     'heightField': None,
                                                                     'heightScale': 'linear',
                                                                     'radiusField': None,
                                                                     'radiusScale': 'linear'}}],
                                      'interactionConfig': {
                                          'tooltip': {'fieldsToShow': {'addresses by polling stations': ['place_uid'],
                                                                       'polling stations outlines': ['place_uid']},
                                                      'enabled': True},
                                          'brush': {'size': 0.5, 'enabled': False}},
                                      'layerBlending': 'normal',
                                      'splitMaps': []},
                         'mapState': {'bearing': 0,
                                      'dragRotate': False,
                                      'latitude': 48.402518445245185,
                                      'longitude': 10.818821808653135,
                                      'pitch': 0,
                                      'zoom': 10.379836309981588,
                                      'isSplit': False},
                         'mapStyle': {'styleType': 'dark',
                                      'topLayerGroups': {},
                                      'visibleLayerGroups': {'label': True,
                                                             'road': True,
                                                             'border': False,
                                                             'building': True,
                                                             'water': True,
                                                             'land': True,
                                                             '3d building': False},
                                      'mapStyles': {}}}}
