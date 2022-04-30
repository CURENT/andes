"""
Support for Jupyter widgets.

Please manually install the following dependencies:

- ipywidgets
- ipysheet

If you are using JupyterLab, do

.. code-block :: bash

    jupyter labextension install @jupyter-widgets/jupyterlab-manager

"""

from collections import OrderedDict

try:
    import ipywidgets as widgets
    from ipywidgets import HBox, VBox
    from IPython.display import display, HTML
except ImportError:
    pass


def edit_system(system):
    """
    Edit a loaded ANDES System with ipywidgets.
    """
    group_models = OrderedDict()

    # works with models with devices only
    model_tabs = dict()
    for group_name, instances in system.groups.items():
        model_names = list(instances.models.keys())
        exist_models = [mname for mname in model_names if system.models[mname].n]

        if len(exist_models):
            group_models[group_name] = exist_models

            for m in group_models[group_name]:
                model_tabs[m] = edit_sheet(system, m)

    # inner tabs
    group_table = OrderedDict()
    for group_name in group_models:
        model_names = group_models[group_name]
        group_table[group_name] = widgets.Tab()
        group_table[group_name].titles = group_models[group_name]  # appears not working
        group_table[group_name].children = [model_tabs[m] for m in model_names]
        for ii in range(len(model_names)):
            group_table[group_name].set_title(ii, group_models[group_name][ii])

    # outer tabs
    sys_tab = widgets.Tab()
    sys_tab.titles = list(group_models.keys())
    sys_tab.children = list(group_table.values())

    for ii in range(len(group_models)):
        sys_tab.set_title(ii, sys_tab.titles[ii])

    return sys_tab


def edit_sheet(system, model: str):
    """
    Use ipysheet to edit parameters of one model.
    """

    sh = system.to_ipysheet(model, vin=True)
    header = widgets.Output()
    output = widgets.Output()

    button_upd = widgets.Button(description="Update")
    button_upd.on_click(on_update)
    button_upd.system = system
    button_upd.model = model
    button_upd.sheet = sh
    button_upd.output = output

    button_close = widgets.Button(description="Close")
    button_close.on_click(on_close)
    button_close.objects = [header, sh, button_upd, button_close, output]

    hbox = HBox((button_upd, button_close))
    ret = VBox((header, sh, hbox, output))

    return ret


def on_update(b):
    """
    Callback for the Update button. Sets new parameters back to System.
    """

    with b.output:
        b.system.from_ipysheet(b.model, b.sheet)
        if not hasattr(b, "label"):
            label = display(HTML("%s: parameter update was successful." % b.model), display_id=True)
            b.label = label
        else:
            b.label.update(HTML("%s: parameter update was successful." % b.model))


def on_close(b):
    """
    Callback for the Close botton. Closes ipywidget objects.
    """

    for item in b.objects:
        item.close()
