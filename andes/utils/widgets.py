"""
Support for Jupyter widgets.
"""

import ipywidgets as widgets

from ipywidgets import VBox, HBox


def edit_system(system):
    """
    Edit a loaded ANDES System with ipywidgets.
    """
    pass


def edit_sheet(system, model: str):
    """
    Use ipysheet to edit parameters of one model.
    """

    sh = system.to_ipysheet(model, vin=True)
    header = widgets.Output()
    with header:
        print(model)

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
        print("%s: parameter update was successful." % b.model)


def on_close(b):
    """
    Callback for the Close botton. Closes ipywidget objects.
    """

    for item in b.objects:
        item.close()
