# Tools for dealing with the raw, unadulterated turning dataframe

import pandas as pd
import numpy as np
from tetromate_webserver import ExperimateLogger
from typing import List

def annotate_cumulative_depth(df, zero_entrances=True, mm_per_turn=None, **kws):

    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, ExperimateLogger):
        mm_per_turn = df.get_mm_per_turn()
        kws.update({'mm_per_turn':mm_per_turn})
        df = df.df
    else:
        raise TypeError("df must be a dataframe or an ExperimateLogger")
    kws.update({'zero_entrances':zero_entrances})

    tetrodes = df.tetrode.unique()
    DF, estimate_entrance, has_entrance = [], [], []
    for t, tetrode in enumerate(tetrodes):

        if np.isnan(tetrode) or tetrode < 0:
            DF.append(None)
            continue

        DF.append(
            _annotate_cumulative_one_tet_depth(df.query(f"tetrode == {tetrode}"), **kws)
        )

        no_entrance = (DF[-1].intent != "entrance").all()
        if no_entrance:
            estimate_entrance.append(t)
        else:
            has_entrance.append(t)


    return DF


def _annotate_one_tet_depth(d, mm_per_turn=None):

    d.loc[:,'turns'] = d.turns.fillna(value=0)
    d.loc[:,'depth'] = d.turns.cumsum()
    d.loc[:, 'area'] = d.area.ffill()

    if mm_per_turn:
        d.loc[:,'depth_mm'] = d.depth * mm_per_turn/12

    entrances = (d.intent == "entrance")
    if zero_entrances and entrances.any():
        entrance_location = np.nonzero(entrances.values)[0].max()
        d.loc[:, 'entrance_depth'] = d.iloc[entrance_location].depth
        if mm_per_turn:
            d.loc[:, 'entrance_depth_mm'] = d.iloc[entrance_location].depth_mm

    return d
