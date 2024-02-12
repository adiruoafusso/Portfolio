import os
import pathlib
import io
import zipfile
import time
import atexit
from flask import request, render_template, session, redirect, url_for, jsonify, send_file
from utils.psomap_tasks_manager import app, build_tracing_map_task, tracing_map_task_status, pd
from utils.psomap_utilities import delete_session_data, unzip_cities_files
from config.config_file import csv_path
import json

# Todo commit results (run one test before committing)

########################################################################################################################
#                                               MAIN VIEW                                                              #
########################################################################################################################


@app.route('/')
def index():
    return render_template('index.html')

########################################################################################################################
#                                                 TASKS                                                                #
########################################################################################################################


tasks_data = {}


@app.route('/status/<list:tasks_id>')
def get_task_status(tasks_id):
    json_list = []
    for task_id in tasks_id:
        city_label = tasks_data[task_id]
        task = build_tracing_map_task.AsyncResult(task_id)
        task_response = tracing_map_task_status(task, city_label)  # clean function if statements
        json_list.append(json.dumps(task_response))
    return json.dumps({'results': json_list})


@app.route('/_get_map', methods=['GET', 'POST'])
def get_map():
    file = request.files['csv_file']
    city_label, file_extension = file.filename.split('.')

    if file_extension == 'zip':
        cities_dict = unzip_cities_files(file)
    else:
        addresses_df = pd.read_csv(file, sep=';') if file_extension in ['csv', 'txt'] else pd.read_excel(file, sep=';')
        addresses_df.reset_index(inplace=True)
        cities_dict = {city_label.capitalize(): addresses_df}

    for city_name, city_df in cities_dict.items():
        task = build_tracing_map_task.delay(city_df.to_json(), city_name)
        tasks_data[task.id] = city_name
        #time.sleep(0.5)

    tasks_ids = list(tasks_data.keys())
    print(tasks_ids)

    return jsonify({}), 202, {'Location': url_for('get_task_status', tasks_id=tasks_ids)}

########################################################################################################################
#                                               MAPPING DATA                                                           #
########################################################################################################################


@app.route('/display_map/<city_label>', methods=['GET'])
def display_map(city_label):
    return redirect(url_for('static', filename='maps_data/{0}/{0}_polling_stations_outlines.html'.format(city_label)))

########################################################################################################################
#                                               DATA ANALYSIS                                                          #
########################################################################################################################


@app.route('/display_table/<city_label>', methods=['GET', 'POST'])
def display_table(city_label):
    addresses_not_found_df = pd.read_csv(csv_path+'{}_addresses_not_found.csv'.format(city_label), sep=';')
    return render_template('analysis.html', tables=[addresses_not_found_df.to_html(classes='data', header="true")])


@app.route('/display_histogram_graph/<city_label>', methods=['GET'])
def display_histogram_graph(city_label):
    return redirect(url_for('static', filename='graph/{}_polling_stations_histogram.html'.format(city_label)))


@app.route('/display_donut_graph/<city_label>', methods=['GET'])
def display_donut_graph(city_label):
    return redirect(url_for('static', filename='graph/{}_geocoding_ratio_donut.html'.format(city_label)))

########################################################################################################################
#                                                   EXPORT                                                             #
########################################################################################################################


@app.route('/export/<city_label>', methods=['GET'])
def export_map(city_label):
    map_data_path = pathlib.Path('./static/maps_data/{}/'.format(city_label))
    time_as_str = time.strftime("%Y/%m/%d-%H:%M:%S")
    zip_filename = "{}_export_{}.zip".format(time_as_str, city_label)
    map_data = io.BytesIO()
    with zipfile.ZipFile(map_data, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        rootdir = os.path.basename(map_data_path)
        for root, dirs, files in os.walk(map_data_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                parent_path = os.path.relpath(file_path, map_data_path)
                arcname = os.path.join(rootdir, parent_path)
                zip_file.write(file_path, arcname)
    map_data.seek(0)
    return send_file(map_data, attachment_filename=zip_filename, as_attachment=True)


if __name__ == '__main__':
    # Schedule files auto-deletion when a session is ended
    atexit.register(delete_session_data)
    app.run(debug=True)
