"""
A pert file template.
"""


def pert(t, system):
    """
    Perturbation function called at each step.

    The function is named "pert" and takes two arguments:
    ``t`` for simulation time, and ``system`` for the system object.

    If the event involves switching, which will create a change
    in the system Jacobian pattern, the ``custom_event`` flag
    needs to be set by

    .. code-block :: python

        self.custom_event = False

    The flag should only be set for the time instant when the
    event is triggered.

    """
    pass
