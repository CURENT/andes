class RoutineBase(object):
    """
    Base class for Routines
    """

    def run(self):
        """
        Entry function for power flow routine

        Returns
        -------
        bool
            Success flag
        """

        raise(NotImplementedError, 'Must be overloaded by routines')

    def reset(self):
        """
        Reset internal states of the routine

        Returns
        -------
        None
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

    def pre(self):
        """
        Pre-check for routine

        Returns
        -------
        bool
            Success flag
        """
        pass

    def post(self):
        """
        Post processing for routine

        Returns
        -------
        bool
            Success flag
        """
        pass
