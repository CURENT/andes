import andes


def test_numparam_online(caplog):
    """
    Ensure In one model, issue a warning only once for NumParam value checks.
    """
    case = andes.get_case("GBnetwork/GBnetwork.m")
    mpc = andes.io.matpower.m2mpc(case)

    mpc['branch'][2:5, 3] = 0

    ss = andes.system.System()
    with caplog.at_level("WARNING"):
        andes.io.matpower.mpc2system(mpc, ss)
        assert caplog.text.count("Non-zero parameter Line.x corrected to 1e-08.") == 1
