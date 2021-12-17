from fipie import VolatilityParity


def test_repr():
    obj = VolatilityParity()
    assert str(obj) == 'VolatilityParity(target_vol=0.1, fully_invested=False)'
