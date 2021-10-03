
def compute_turns_to_distance(EL:object, areas:list, desired_distance:float):
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

def compute_tetrode_distance_difference(tet1, tet2, DF=None, prop='depth_mm'):

    from .notes import get_notes
    note1 = get_notes(tet1, DF).reset_index('depth_mm')[prop]
    note2 = get_notes(tet2, DF).reset_index('depth_mm')[prop]
    note1 = note1[note1 != ''].astype('float').iloc[-1]
    note2 = note2[note2 != ''].astype('float').iloc[-1]

    return (note2 - note1)
