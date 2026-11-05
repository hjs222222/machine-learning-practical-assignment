from pathlib import Path
import pandas as pd
from data import load_raw
from features import build_features
from data_analysis import field_table, correlation_frame


def test_analysis_covers_all_fields_and_lagged_factors():
    raw = load_raw()
    table = field_table()
    assert set(table.field) == {'Date',*raw.columns}
    frame = correlation_frame(raw)
    assert all(c in frame for c in ('Discharge','Discharge_lag1','Prcp_lag1','Tmax_lag1'))
    assert not raw.Swe.any()
