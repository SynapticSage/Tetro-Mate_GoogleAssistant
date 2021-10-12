# Tools for dealing with the raw, unadulterated turning dataframe

import pandas as pd
import numpy as np
from tetromate_webserver import ExperimateLogger
from typing import List, Union


def get_last_depth(df:Union[list, pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]):
    '''
    '''
    if isinstance(df, pd.DataFrame) and "intent" not in df:
        grab_last = (lambda d: d.sort_values('depth').iloc[-1] if d is not None else None)
    else:
        grab_last = (lambda d: d.query('intent == "adjust-tetrode"')
                                .sort_values('depth').iloc[-1] if d is not None else None)
    if isinstance(df, list):
        return pd.concat([grab_last(d) for d in list])
    elif isinstance(df, pd.DataFrame):
         return get_last_depth(df.groupby('tetrode'))
    else:
         return df.apply(grab_last)


def annotate_cumulative_depth(df, zero_entrances=True, mm_per_turn=None,
                              **kws):

    kws.update({'mm_per_turn':mm_per_turn})
    kws.update({'zero_entrances':zero_entrances})

    tetrodes = df.tetrode.unique()
    DF, estimate_entrance, has_entrance = [], [], []
    for t, tetrode in enumerate(tetrodes):

        if np.isnan(tetrode) or tetrode < 0:
            DF.append(None)
            continue
        DF.append(
            _annotate_one_tet_depth(df.query(f"tetrode == {tetrode}"), **kws)
        )

        no_entrance = (DF[-1].intent != "entrance").all()
        if no_entrance:
            estimate_entrance.append(t)
        else:
            has_entrance.append(t)

    return DF


def annotate_cumulative_depth(df, zero_entrances=True, mm_per_turn=None, **kws):

    kws.update({'mm_per_turn':mm_per_turn})
    kws.update({'zero_entrances':zero_entrances})

    tetrodes = df.tetrode.unique()
    DF, estimate_entrance, has_entrance = [], [], []
    for t, tetrode in enumerate(tetrodes):

        if np.isnan(tetrode) or tetrode < 0:
            DF.append(None)
            continue
        DF.append(
            _annotate_one_tet_depth(df.query(f"tetrode == {tetrode}"), **kws)
        )

        no_entrance = (DF[-1].intent != "entrance").all()
        if no_entrance:
            estimate_entrance.append(t)
        else:
            has_entrance.append(t)


    return DF


def _annotate_one_tet_depth(d, mm_per_turn=None, zero_entrances=False):

    d.loc[:,'turns'] = d.turns.fillna(value=0)
    d.loc[:,'depth'] = d.turns.cumsum()
    d.loc[:, 'area'] = d.area.ffill()

    if mm_per_turn:
        d.loc[:,'depth_mm'] = (d.depth *
                               mm_per_turn/12)

    entrances = (d.intent == "entrance")
    if zero_entrances and entrances.any():
        entrance_location = np.nonzero(entrances.values)[0].max()
        d.loc[:, 'entrance_depth'] = d.iloc[entrance_location].depth
        d.loc[:, 'entrance_depth'] = d.loc[:,'entrance_depth'].ffill()
        if mm_per_turn:
            d.loc[:, 'entrance_depth_mm'] = d.iloc[entrance_location].depth_mm
        d.loc[:, 'entrance_depth_mm'] = d.loc[:,'entrance_depth_mm'].ffill()

    return d

def append_properties(df:Union[pd.DataFrame, list], prop_frame:pd.DataFrame) -> pd.DataFrame:
    '''
    '''

    if isinstance(df, list):
        df = pd.concat(df)

    return df.merge(prop_frame, on="tetrode", suffixes=('','_prop'))
