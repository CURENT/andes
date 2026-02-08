"""
GLODF Test Code

Test to ensure IEEE 14 bus system has less than 5% error with the following
line outages:
    ['Line_1', 'Line_3', 'Line_5', 'Line_7', 'Line_9',]
"""

from andes.routines.glodf import GLODF
import andes
import numpy as np
import pandas as pd

def test_flow_after_glodf():
    """
    Example code to test the GLODF class
    """
    print("GLODF Test")
    # load system
    ss = andes.load(andes.get_case("ieee14/ieee14_linetrip.xlsx"))
    
    # solve system
    ss.PFlow.run()
    
    # create GLODF object
    g = GLODF(ss)
 
    # lines to be taken out
    change_lines = ['Line_1', 'Line_3', 'Line_5', 'Line_7', 'Line_9',]
    
    lines_before = np.copy(ss.Line.a1.e)
    lines_after = g.flow_after_glodf(change_lines)
    
    np.set_printoptions(precision=5)
    
    uid_lines = np.atleast_1d(ss.Line.idx2uid(change_lines))
    # turn off lines and resolve1
    for i in range(np.size(change_lines)):
        ss.Line.u.v[uid_lines] = 0
    ss.PFlow.run()
    
    lineflows = {"bus1": ss.Line.bus1.v,
                 "bus2": ss.Line.bus2.v,
                 "P1 before": lines_before,
                 "P1 GLODF": lines_after,
                 "P1 re-solved": ss.Line.a1.e,
                 "error": np.abs(lines_after - ss.Line.a1.e) }
    
    df_lineflow = pd.DataFrame(lineflows, index=ss.Line.idx.v)
    
    print(df_lineflow)
    
    percent_error = np.mean(np.abs((ss.Line.a1.e - lines_after))) / np.mean(np.abs(ss.Line.a1.e)) * 100
    print("normalized absolute percent error: {:.3f}%".format(percent_error))
    
    # Check if lines_after is close to ss.Line.a1.e
    assert np.allclose(percent_error, 0, atol=5) # error should be less than 5%
    
if __name__ == "__main__":
    test_flow_after_glodf()
