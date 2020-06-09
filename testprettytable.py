#  ____           _   _           _____     _     _      
# |  _ \ _ __ ___| |_| |_ _   _  |_   _|_ _| |__ | | ___ 
# | |_) | '__/ _ \ __| __| | | |   | |/ _` | '_ \| |/ _ \
# |  __/| | |  __/ |_| |_| |_| |   | | (_| | |_) | |  __/
# |_|   |_|  \___|\__|\__|\__, |   |_|\__,_|_.__/|_|\___|
#                         |___/                          
# Preamble
grouping_name = 'day'
grouping = rawdf.index.day

notation_intents = set(rawdf.columns) - {"adjust-tetrode", "marker"}

## UNIFIED ENTRY PER ADJUSTMENT ##
## -------------------------------

# If markers are present, fill forward (for later!)
markersPresent = 'marker' in rawdf.columns and not df.marker.isnull().all()
# Add the current marker to each
if markersPresent:
    rawdf['marker'] = rawdf.marker.ffill()
# Mark each note by the adjustment it belongs to
rawdf['adjustment'] = rawdf.intent == 'adjust-tetrode'
rawdf['adjustment'] = rawdf.adjustment.cumsum()
def collapse_notes(frame):
    adjust_tetrode_portion = frame.loc[frame.intent == 'adjust-tetrode']
    notes_portion = frame.loc[frame.intent != 'adjust-tetrode']
    id_vars = notes_portion.columns.intersection(['datetime','marker','tetrode','adjustment','intent'])
    notes_portion = notes_portion.reset_index().melt(id_vars=id_vars, value_vars=set(notes_portion.columns)-set(id_vars), 
               var_name='type', value_name='value')
    notes_portion = notes_portion.dropna(how='any')
    notes = notes_portion.intent + "_" +  notes_portion.type + "=>" + notes_portion.value.astype('str')
    notes = "\n".join(notes.values.tolist())
    adjust_tetrode_portion.loc[:,'notes'] = notes
    adjust_tetrode_portion.drop(columns='note',inplace=True)

    return adjust_tetrode_portion

# Collapse notes by adjustment
tetrodeAdjustments = (rawdf
                      .groupby('adjustment')
                      .apply(collapse_notes)
                      .dropna(axis=1, how='all')
                      .reset_index(['adjustment'], drop=True)
                      .reset_index()
                      .set_index(['datetime'])
                      )
tetrodeAdjustments


# Figure out times of the tetrode
#tetrodeAdjustments     = rawdf[rawdf.intent=='adjust-tetrode']
#tetrodeAdjustmentTimes = rawdf.index

## ANNOTATE (including the grouping)
## ---------------------------------

# Create the depth
tetrodeAdjustments.loc[:,'depth'] = tetrodeAdjustments.groupby('tetrode').turns.cumsum()

# Label with grouping
tetrodeAdjustments.loc[:, grouping_name] = tetrodeAdjustments.index.day

# Determine the super group, which is the cartesian product of marker and grouping
if markersPresent:
    tetrodeAdjustments.loc[:,'supergroup'] = (tetrodeAdjustments[grouping_name].astype('str') + ' - ' + tetrodeAdjustments['marker'].astype('str'))
    grouping = tetrodeAdjustments['supergroup']

## PIVOT
## -----
tetrodeAdjustments = (tetrodeAdjustments
 .drop(columns=[x for x in tetrodeAdjustments.columns if 'level_' in x])
 .reset_index()
 .drop(columns=[x for x in tetrodeAdjustments.columns if 'level_' in x])
  )
pretty_table = []
grouping = np.unique(grouping, return_inverse=True)[1]
for group, data in tetrodeAdjustments.groupby(grouping):
    print(f'Creating table for group = {group}')
    assert((data.day == data.day.iloc[0]).all())
    # B. Reindex by the ordinal position within group per tetrode
    count_ordinal = lambda x : pd.Series(np.arange(len(x))+1, index=x.index)
    data = data.sort_values(['tetrode','datetime'])
    ordinals = pd.DataFrame(data.groupby('tetrode').apply(count_ordinal))
    ordinals.set_index(data.index, inplace=True)
    data['ordinal'] = ordinals
    
    #data = data.reset_index().set_index(['day','marker','ordinal'])
    # C. PIVOT
    data = (data
            .pivot_table(index=['day','marker','ordinal'], 
                         columns=['tetrode'],
                         values=['turns','depth','notes'])
            .swaplevel(0,1,axis=1)
            .sort_index(axis=1)
            .fillna('')
            )
    pretty_table.append(data)

pretty_table = pd.concat(pretty_table, axis=0)
