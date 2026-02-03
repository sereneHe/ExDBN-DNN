# Renamed from test_shd_utils.py
import pytest

from dagsolvers.shd_utils import *


@pytest.fixture
def b_true():
    return np.zeros((4, 4))


@pytest.fixture
def b_est():
    return np.zeros((4, 4))

@pytest.mark.parametrize("e", [0, 1, 2, -1, -2, -3])
def test_shd1(b_true, b_est, e):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    b_true[1, 0] = e
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 0

@pytest.mark.parametrize("e", [1, 2, -1, -2, -3])
def test_shd2(b_true, b_est, e):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 1


@pytest.mark.parametrize("e", [1, 2, -1, -2, -3])
@pytest.mark.parametrize("f", [1, 2, -1, -2, -3])
def test_shd3(b_true, b_est, e, f):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    b_true[1, 0] = f
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    if e == f:
        true_shd = 0
    elif e in [1, 2, -3] and f in [1, 2, -3]:
        true_shd = 0
    elif e in [-1, 2, -3] and f in [-1, 2, -3]:
        true_shd = 0
    elif e in [-2, -3] and f in [-2, -3]:
        true_shd = 0
    elif e != f:
        true_shd = 0.5
    else:
        true_shd = 1
    assert shd == true_shd

@pytest.mark.parametrize("e", [1, 2, -1, -2, -3])
@pytest.mark.parametrize("f", [1, 2, -1, -2, -3])
def test_shd4(b_true, b_est, e, f):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    b_true[0, 1] = f
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    if e != f:
        if e in [1, -3] and f in [1, -3]:
            true_shd = 0
        elif e in [2,-1,3] and f in [2,-1, -3]:
            true_shd = 0
        elif e in [-2, -3] and f in [-2, -3]:
            true_shd = 0
        elif e in [2, -3] and f in [2, -3]:
            true_shd = 0
        elif e in [-1, -3] and f in [-1, -3]:
            true_shd = 0
        else:
            true_shd = 0.5
    else:
        if e == 1:
            true_shd = 0.5
        else:
            true_shd = 0

    assert shd == true_shd

def test_shd5(b_true, b_est):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = 1
    b_true[0, 1] = 1
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 0.5

def test_shd6(b_true, b_est):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = 1
    b_true[1, 0] = 1
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 0

def test_shd7(b_true, b_est):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = 1
    b_true[1, 0] = 0
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 1
# Renamed from test_shd_utils.py
import pytest

from dagsolvers.shd_utils import *


@pytest.fixture
def b_true():
    return np.zeros((4, 4))


@pytest.fixture
def b_est():
    return np.zeros((4, 4))

@pytest.mark.parametrize("e", [0, 1, 2, -1, -2, -3])
def test_shd1(b_true, b_est, e):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    b_true[1, 0] = e
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 0

@pytest.mark.parametrize("e", [1, 2, -1, -2, -3])
def test_shd2(b_true, b_est, e):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 1


@pytest.mark.parametrize("e", [1, 2, -1, -2, -3])
@pytest.mark.parametrize("f", [1, 2, -1, -2, -3])
def test_shd3(b_true, b_est, e, f):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    b_true[1, 0] = f
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    if e == f:
        true_shd = 0
    elif e in [1, 2, -3] and f in [1, 2, -3]:
        true_shd = 0
    elif e in [-1, 2, -3] and f in [-1, 2, -3]:
        true_shd = 0
    elif e in [-2, -3] and f in [-2, -3]:
        true_shd = 0
    elif e != f:
        true_shd = 0.5
    else:
        true_shd = 1
    assert shd == true_shd


@pytest.mark.parametrize("e", [1, 2, -1, -2, -3])
@pytest.mark.parametrize("f", [1, 2, -1, -2, -3])
def test_shd4(b_true, b_est, e, f):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = e
    b_true[0, 1] = f
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    if e != f:
        if e in [1, -3] and f in [1, -3]:
            true_shd = 0
        elif e in [2,-1,3] and f in [2,-1, -3]:
            true_shd = 0
        elif e in [-2, -3] and f in [-2, -3]:
            true_shd = 0
        elif e in [2, -3] and f in [2, -3]:
            true_shd = 0
        elif e in [-1, -3] and f in [-1, -3]:
            true_shd = 0
        else:
            true_shd = 0.5
    else:
        if e == 1:
            true_shd = 0.5
        else:
            true_shd = 0

    assert shd == true_shd

def test_shd5(b_true, b_est):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = 1
    b_true[0, 1] = 1
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 0.5

def test_shd6(b_true, b_est):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = 1
    b_true[1, 0] = 1
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 0

def test_shd7(b_true, b_est):
    b_true = b_true.copy()
    b_est = b_est.copy()
    b_est[1, 0] = 1
    b_true[1, 0] = 0
    shd, *_ = calculate_shd(b_true, b_est, [], [])
    assert shd == 1
