import pandas as pd
import numpy as  np

class OldFunctions:
    '''
    Groups together older methods
    '''

    @staticmethod
    def compute_turns_to_distance(EL:object, areas:list, desired_distance:float,
                                  minus_entrance:bool=True) -> dict:
        '''
        Takes a tetromate object and list of areas and a desired distance, and then
        computes how many turns in multiple different units of turns to reach the
        desired distance. Returns pandas dataframes instructing the user how many
        turns per electrode bundle.
        '''


        area_query = [f"area == '{area}'" for area in areas]
        area_query = " or ".join(area_query)

        df = EL.get_current_property_table(properties=["depth_mm"])[-1]

        mm_required                                   = (desired_distance
                       - (df.reset_index('area') .query(area_query) .set_index('area', append = True) .astype('float') + EL.const_depth_mm))

        if minus_entrance:
            pass

        fullturns_required = (EL.get_turns_per_mm() * mm_required.copy()).rename(columns={'depth_mm':'fullturn'})
        twelths_required   = (fullturns_required.copy() * 12).rename(columns={'fullturn':'twelths'})
        quarters_required  = (fullturns_required.copy() * 4).rename(columns={'fullturn':'quarters'})

        move_dict ={
            'mm'       : mm_required,
            'fullturn' : fullturns_required,
            'twelths'  : twelths_required,
            'quarters' : quarters_required,
        }

        return move_dict

    @staticmethod
    def compute_tetrode_distance_difference(tet1, tet2, DF=None, prop='depth_mm'):

        from .notes import get_notes
        note1 = get_notes(tet1, DF).reset_index('depth_mm')[prop]
        note2 = get_notes(tet2, DF).reset_index('depth_mm')[prop]
        note1 = note1[note1 != ''].astype('float').iloc[-1]
        note2 = note2[note2 != ''].astype('float').iloc[-1]

        return (note2 - note1)


    @staticmethod
    def compute_entrances_per_tetrode(EL):
        pass



def compute_turns_to_distance(EL:object, areas:list, desired_distance:float,
                              minus_entrance:bool=True) -> dict:

    from . import raw
    area_query = [f"area_prop == '{area}'" for area in areas]
    area_query = " or ".join(area_query)

    df = raw.annotate_cumulative_depth(EL.df,
                                       mm_per_turn=EL.get_mm_per_turn())

    df = pd.concat(df)

    assert(not df.entrance_depth.isnull().all())


    # Current way annotations work : TODO can be less awkward than a google sheet fetch
    EL.fetch_tetrode_properties()
    df = raw.append_properties(df, EL.tetrode_properties)

    # Extract dpeth only data
    if areas:
        depth_data = df.query(area_query)
    else:
        depth_data = df.copy()

    depth_data = (depth_data
                 .set_index(['tetrode','area_prop'], append=True)[['depth','depth_mm','entrance_depth_mm','entrance_depth','intent']])
    depth_data = depth_data.reset_index()
    depth_data = raw.get_last_depth(depth_data)

    depth_data = depth_data.reset_index(drop=True).set_index(['area_prop','tetrode']).sort_index()
    if minus_entrance:
        depth_data = (depth_data['depth_mm']
                      - depth_data['entrance_depth_mm'])
    else:
        depth_data = depth_data['depth_mm']


    if desired_distance is not None:
        mm_required = (desired_distance
                       - (depth_data + EL.const_depth_mm))
    else:
        mm_required = depth_data + EL.const_depth_mm

    mm_required = pd.DataFrame(mm_required)

    fullturns_required = (EL.get_turns_per_mm() * mm_required.copy()).rename(columns={0:'fullturn'})
    twelths_required   = (fullturns_required.copy() * 12).rename(columns={'fullturn':'twelths'})
    quarters_required  = (fullturns_required.copy() * 4).rename(columns={'fullturn':'quarters'})

    move_dict ={
        'mm'       : mm_required,
        'fullturn' : fullturns_required,
        'twelths'  : twelths_required,
        'quarters' : quarters_required,
    }

    return move_dict


def distance_to_area(EL, turn_style='fullturn', method='mean',
                     target_area=None, guessna=True, *pos, **kws):
    '''
    Obtains distances of tetrodes in quater/halves/fullturns to
    a target_area set of tetrodes (their mean or median).
    '''

    prop = compute_turns_to_distance(EL, *pos, **kws)

    if method == 'mean':
        meth = lambda x : x.mean()
    else:
        meth = lambda x : x.median()


    turns = meth(prop[turn_style].groupby(['area_prop'])).loc[target_area] - prop[turn_style]

    if guessna:
        turns = turns.fillna(meth(prop[turn_style].groupby(['area_prop'])))

    return turns
