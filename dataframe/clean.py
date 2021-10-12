import pandas as pd

def clean_output(df:pd.DataFrame):
    '''
    Cleans df before outputting to a google sheet
    '''
    df = df.astype({'tetrode':str, 'turns':str, 'magnitude':str})
    columns = [col for col in df.columns if "Unnamed" not in col]
    df = df[columns].applymap(lambda x: x if x != 'nan' else '')
    if 'index' in df:
        df.drop(columns='index', inplace=True)
    #df.index = df.index.astype('str')
    return df


def clean_raw(df, *pos, **kws):
    '''

    '''
    from . import version

    # Version corrections
    df = version.ready(df)

    # DTYPE corrections
    df = set_types(df, *pos, **kws)
    dtypes = df.dtypes
    for field in ['tetrode', 'turns', 'magnitude']:
        if dtypes[field] == 'O':
            df.loc[:, field] = df.loc[:, field].astype(float)

    return df

# TYPE - CHECKING SUBFUNCTIONS


def set_types(df, continuous_explode):
    '''
    Set types of our tracker dataframe that keeps tabs on all
    commands the user submits

    If df is provided, this works like a static function!
    '''
    import numpy as np
    import pytz

    def klugey_datetime_correction(df):
        try:
            index = df.index
            if not isinstance(index, pd.DatetimeIndex):
                index = pd.DatetimeIndex(index)
            index = index.dt.tz_convert(pytz.timezone('US/Eastern'))
            index = pd.DatetimeIndex(index)
            index.name = 'datetime'
            df = df.set_index(index)
        except Exception as E:
            try:
                index = df.index
                if not isinstance(index, pd.DatetimeIndex, utc=True):
                    index = pd.DatetimeIndex(index)
                #index = index.dt.tz_convert('EST')
                index = index.tz_convert('EST')
                index = pd.DatetimeIndex(index)
                index.name = 'datetime'
                df = df.set_index(index)
            except Exception:
                index = df.index
                if (not isinstance(index, pd.DatetimeIndex) and not
                        isinstance(index.values[0], pd._libs.tslib.Timestamp)):
                    index = pd.DatetimeIndex(index)
                index.name = 'datetime'
                df = df.set_index(index)

        return df

    # Make index pd.Datetime!
    if "datetime" in df.columns:
        index = pd.to_datetime(df.datetime, utc=True)
        index = index.dt.tz_convert(pytz.timezone('US/Eastern'))
        index.name = 'datetime'
        df = df.drop(columns='datetime').set_index(index)
    else:
        if df.index.name != "datetime":
            print("Index not datetime")
            df = klugey_datetime_correction(df)
        if not hasattr(df.index, 'day'):
            print("Index does not have day!")
            df = klugey_datetime_correction(df)

    # Determine tetrode type based on user settings
    if continuous_explode:
        tetrode_type = np.double
    else:
        tetrode_type = np.object

    # Retype
    df = df.astype({x:y for x,y in {'tetrode':tetrode_type,
                                    'turns':np.double,
                                    'magnitude':np.double,
                                    'adjustment':np.double,
                                    'depth':np.double}.items()
                          if x in df.columns})

    return df
