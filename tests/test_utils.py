from netreslib import utils
import re
import numpy as np

def test_parser():
    for i in ["1", "0", "+1000", "-3", "-0"]:
        assert re.match(rf"^{utils.INT_RE}$", i)
    for i in ["1", "0", "+1000", "-3", "-0", ".0", "-inf", ".1e-3", "1.e+30"]:
        assert re.match(rf"^{utils.FLOAT_RE}$", i)

    assert utils.parse_generator(" 1 ")() == 1
    assert utils.parse_generator("+1.0 ")() == 1.0
    assert utils.parse_generator(" -.0")() == -0.0

    g = utils.parse_generator("1 .. 5")
    assert set([g() for i in range(100)]) == set([1, 2, 3, 4])
    g = utils.parse_generator(" 1,2 , 3")
    assert set([g() for i in range(100)]) == set([1, 2, 3])
    g = utils.parse_generator("1, 2,-3e1")
    assert set([g() for i in range(100)]) == set([1.0, 2.0, -30.0])

    g = utils.parse_generator(" u(1,2)")
    a = np.array([g() for i in range(100)])
    assert all(a >= 1.0) and all(a < 2.0) and sum(a > 1.5) > 40 and sum(a > 1.5) < 60
    g = utils.parse_generator("LU ( 1 , 10) ")
    a = np.array([g() for i in range(100)])
    assert all(a >= 1.0) and all(a < 10.0) and sum(a > 3.16) > 40 and sum(a > 3.16) < 60
