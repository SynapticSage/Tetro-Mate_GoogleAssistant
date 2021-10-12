# For updating older versions of the dataframe to newer versions

from tetromate_webserver import ExperimateLogger

from typing import Union
import pandas as pd
import numpy as np

def ready(df):
    entrances = df.query('intent == "entrance"')
    if entrances.tetrode.isnull().any() or (entrances.tetrode == -1).any():
        df = upgrade_to_tetrode_labeled_entrance(df)
    return df

def upgrade_to_tetrode_labeled_entrance(df:pd.DataFrame,
                                        fix_negative_one_default_values=True):
    '''
    Newer versions annotate the tetrode for the entrance, rather than leaving
    it blank. Blank entries were assumed to have the tetrode of the whichver
    previous entry notated tetrode.
    '''

    df = df.copy()
    df = df.reset_index()

    if fix_negative_one_default_values:
        x = df.loc[:,'tetrode']
        x[df.tetrode == -1] = np.nan

    # Pass tetrodes ahead into nan entries
    df_fillforward = df.copy()
    df_fillforward.tetrode.ffill(inplace=True)

    # Extract entries related to entrances
    chunk = df_fillforward[df_fillforward.intent == "entrance"]
    df.loc[chunk.index, 'tetrode'] = chunk.tetrode

    return df
