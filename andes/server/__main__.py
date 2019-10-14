import sys
import os
import threading  # NOQA
import andes
from andes.utils.math import to_number
import json

try:
    import flask
    from flask import Flask, request  # NOQA
    from flask.json import jsonify  # NOQA
    from flask_restful import Api  # NOQA
except ImportError:
    print("Flask import error. Install optional package `flask`, 'flask_restful' and `requests`")
    sys.exit(1)


app = Flask(__name__)
api = Api(app)


andes.main.config_logger()
systems = {}
sim_thread = {}


@app.route('/status')
def get_status():
    sysid = request.args.get('sysid', None)

    if sysid not in systems:
        return jsonify('0')
    else:
        if systems[sysid].tds.t != systems[sysid].tds.config.tf:
            return jsonify('1')
        else:
            return jsonify('2')


@app.route('/load')
def load():
    default_path = os.getcwd()
    name = request.args.get('name', '')
    path = os.path.join(default_path, name)
    with_dime = request.args.get('with_dime', 0)
    tf = request.args.get('tf', 20)

    n_system = len(systems)
    params = {"case": path, "verbose": 10, "tf": float(tf)}

    if with_dime == '1':
        params.update({"dime": 'ipc:///tmp/dime'})
    if with_dime == '2':
        params.update({"dime": 'ipc:///tmp/dime2'})
    try:
        system_instance = andes.main.run(**params)
        globals()['systems'][str(n_system + 1)] = system_instance

    except FileNotFoundError:
        flask.abort(404)

    return jsonify(len(systems))


@app.route('/unload')
def unload():
    """
    Unload a system

    Returns
    -------

    """
    sysid = request.args.get('sysid', None)
    force = request.args.get('force', False)
    if force == 'True':
        force = True

    if not sysid or sysid not in systems:
        flask.abort(404)

    print('System <{}> unload requested'.format(sysid))

    if sysid in sim_thread:
        if force:
            sim_thread[sysid].join(1)
        else:
            sim_thread[sysid].join()

    # while sim_thread[sysid].isAlive():
    #     pass
    #
    # sim_thread.pop(sysid)
    # systems.pop(sysid)

    print('System <{}> join called successfully'.format(sysid))

    return jsonify({'response': 'success'})


@app.route('/run')
def run():
    sysid = request.args.get('sysid', None)

    if sysid not in systems:
        flask.abort(400)

    system = systems[sysid]

    if request.method == "GET":

        if sysid in sim_thread:
            if sim_thread[sysid].isAlive():
                flask.abort(500)
            else:
                sim_thread.pop(sysid)

        system.pflow.run()
        system.tds.init()

        system.tds.config.qrt = True

        thread = threading.Thread(target=system.tds.run)
        sim_thread[sysid] = thread
        thread.start()

    return jsonify({'response': 'success'})


@app.route('/param', methods=['GET', 'POST'])
def get_model_param():

    args_json = request.args.get('args_json', '')

    args = json.loads(args_json)

    sysid = str(args.get('sysid', None))
    model_name = args.get('name', None)
    var_name = args.get('var', None)
    idx = args.get('idx', None)
    value = args.get('value', None)
    sysbase = args.get('sysbase', 'False')

    if not sysid:
        flask.abort(400)
    if sysbase == 'False':
        sysbase = False
    elif sysbase == 'True':
        sysbase = True

    system = systems[sysid]

    if request.method == 'GET':

        # if idx is not None:
        #     idx = to_number(idx)

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
                        return jsonify(list(model_ref.get_field(field=var_name)))

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
            print("var_name {} not exist".format(var_name))
            flask.abort(404)

        if sysbase == 'False':
            sysbase = False
        elif sysbase == 'True':
            sysbase = True

        if isinstance(idx, str):
            if idx[0] == '[' and idx[-1] == ']':
                idx = [float(i) for i in idx[1:-1].strip(',')]

        if isinstance(value, str):
            if value[0] == '[' and value[-1] == ']':
                value = [float(i) for i in value[1:-1].strip(',')]

        model_ref.set_field(var_name, idx, value, sysbase)
        model_ref.reload_new_param()
        return jsonify(list(model_ref.get_field(var_name, idx)))


@app.route('/sim_time')
def get_simulation_time():
    """
    Get the simulation time of a system

    Returns
    -------
    str : time
    """
    sysid = request.args.get('sysid', None)

    if sysid is None:
        flask.abort(400)

    system = systems[sysid]

    return jsonify(system.tds.t)


@app.route('/streaming')
def get_streaming_data():
    """
    Get the latest variable in `varout`

    Returns
    -------

    """
    sysid = request.args.get('sysid', None)
    if sysid is None or sysid not in systems:
        flask.abort(400)

    system = systems[sysid]

    return jsonify(system.varout.get_latest_data())


if __name__ == '__main__':
    app.run(host='192.168.1.200', port='7000')
