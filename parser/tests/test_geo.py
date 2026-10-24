"""Расстояния по координатам (geo.py)."""
import pytest

import geo


def test_distance_zero_for_same_point():
    assert geo.distance_km((55.75, 37.62), (55.75, 37.62)) == pytest.approx(0.0, abs=1e-9)


def test_moscow_to_saint_petersburg_about_630_km():
    d = geo.distance_km((55.7558, 37.6173), (59.9343, 30.3351))
    assert 620 < d < 650


def test_distance_is_symmetric():
    a, b = (56.84, 60.61), (55.03, 82.92)  # Екатеринбург, Новосибирск
    assert geo.distance_km(a, b) == pytest.approx(geo.distance_km(b, a))


def test_city_csv_is_loaded_offline():
    # справочник городов лежит в репозитории: без сети Москва должна находиться
    assert (geo.CSV_PATH).exists()
    static = geo._load_static()
    assert static, "справочник городов пустой"
