class RoutineBase(object):
    """
    Base class for Routines
    """

    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        """
        Entry function for power flow routine

        Returns
        -------
        bool
            Success flag
        """

        raise(NotImplementedError, 'Must be overloaded by routines')

    def report(self):
        """
        Format report from the results

        Returns
        -------
        str
            Output string
        """
        raise(NotImplementedError, 'Must be overloaded by routines')
