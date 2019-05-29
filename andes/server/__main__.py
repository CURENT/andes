import os
import andes
import threading  # NOQA

from andes.utils.math import to_number
import flask
from flask import Flask, request  # NOQA
from flask_restful import Resource, Api  # NOQA
from flask.json import jsonify  # NOQA

app = Flask(__name__)
api = Api(app)


andes.main.config_logger()
system = None
sim_thread = None


@app.route('/load')
def load():
    default_path = os.getcwd()
    name = request.args.get('name', '')
    path = os.path.join(default_path, name)

    try:
        globals()['system'] = andes.main.run(case=path)
    except FileNotFoundError:
        flask.abort(404)

    return jsonify(system.devman.devices)


@app.route('/run')
def run():
    simulation_time = request.args.get('time', 0)
    # routine = request.args.get('routine', None)

    if system is None:
        flask.abort(400)
    if simulation_time == 0:
        flask.abort(400)

    if request.method == "GET":
        system.pflow.run()
        system.tds.init()

        system.tds.config.qrt = True
        system.tds.config.tf = float(simulation_time)

        sim_thread = threading.Thread(target=system.tds.run)
        sim_thread.start()

    return jsonify({'response': 'success'})


@app.route('/param', methods=['GET', 'POST'])
def get_model_param():
    model_name = request.args.get('name', None)
    var_name = request.args.get('var', None)
    idx = request.args.get('idx', None)
    value = request.args.get('value', None)

    if not system:
        flask.abort(400)

    if request.method == 'GET':

        if idx is not None:
            idx = to_number(idx)

        if request.method == 'GET':
            if not model_name or (model_name not in system.devman.devices):
                return 'Model name <{}> invalid or not loaded in system'.format(model_name)

            else:  # with model_name

                model_ref = system.__dict__[model_name]
                if var_name:  # with `var_name`
                    if var_name not in model_ref.__dict__:
                        return 'Error: variable <{}> not exist in <{}>'.format(var_name, model_name)

                    if idx:
                        return jsonify(model_ref.get_field(field=var_name, idx=idx))
                    elif not idx:
                        return jsonify(list(model_ref.__dict__[var_name]))

                elif not var_name:  # without `var_name`
                    if idx:
                        if idx not in model_ref.idx:
                            return 'Error: idx <{}> not exist in <{}>'.format(idx, model_name)
                        return jsonify(model_ref.get_element_data(idx))
                    elif not idx:
                        return jsonify(model_ref.data_to_list())

    elif request.method == 'POST':
        if any([model_name, var_name, idx, value]) is None:
            flask.abort(400)

        model_ref = system.__dict__[model_name]

        if var_name not in model_ref.__dict__:
            flask.abort(404)

        if idx not in model_ref.idx:
            flask.abort(404)

        model_ref.set_field(var_name, idx, value)
        return jsonify(model_ref.get_field(var_name, idx))


@app.route('/sim_time')
def get_simulation_time():
    """
    Get the simulation time of a system

    Returns
    -------
    str : time
    """
    if system is None:
        return 'Error: system not loaded'
    return jsonify(system.tds.t)


@app.route('/streaming')
def get_streaming_data():
    """
    Get the latest variable in `varout`

    Returns
    -------

    """
    if system is None:
        return 'Error: system not loaded'
    return jsonify(system.varout.get_latest_data())


if __name__ == '__main__':
    app.run(port='5000')
