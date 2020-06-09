from flask import Flask, request, jsonify, render_template
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Connecting pandas to google cloud
import arrow
import pandas as pd
import gspread_dataframe as gsd
class ExperimateLogger:
    '''
    ExperimateLogger

    Handles all of the logging functionality of listening for Google Assistant commands
    and logging the tetrode lowering data to the the cloud (Google Sheets). Receives
    webhook data from google and logs them to an internal pandas spreadsheet. It
    then logs that spreadsheet to the cloud.
    '''
    
    ## CONSTRUCTOR
    def __init__(self, credentials, key="", url="", title="", continuous_explode=False,
                 log_pretty_table=False, continuous_cloud_update=False):
        import gspread

        self.continuous_cloud_update = continuous_cloud_update # Whether to continuously reload changes from the cloud
        self.continuous_explode = continuous_explode # Whether to explode tetrode lists in the raw data as we go

        # Find google sheet
        self.google_console = gspread.authorize(credentials)
        if title:
            self.spreadsheet = self.google_console.open(title)
        elif url:
            self.spreadsheet = self.google_console.open_by_url(url)
        elif key:
            self.spreadsheet = self.google_console.open_by_key(key)
        else:
            raise NameError("Must provide key, url, or title of googlespreadsheet")
        
        # Open basic worksheets
        if self.exist_worksheet("Raw"):
            self.raw_worksheet = self.spreadsheet.worksheet("Raw")
        else:
            self.raw_worksheet = self.spreadsheet.add_worksheet("Raw", rows=1, cols=1)
        if self.exist_worksheet("Summary"):
            self.pretty_worksheet = self.spreadsheet.worksheet("Summary")
        else:
            self.pretty_worksheet = self.spreadsheet.add_worksheet("Summary", rows=1, cols=1)

        # Load Internal data
        self.df = gsd.get_as_dataframe(self.raw_worksheet, include_index=True, parse_dates=True, userows=[0,1])
        self.set_types()
        #if "datetime" in self.df.columns:
        #    I = pd.to_datetime(self.df.datetime, utc=True)
        #    I = I.dt.tz_convert('EST')
        #    I = pd.DatetimeIndex(I)
        #    I.name = 'datetime'
        #    self.df = self.df.drop(columns='datetime').set_index(I)

        # If all nans, then we need to initialize
        if self.df.isna().all().all():
            self.df = pd.DataFrame([], 
                                   columns=['intent','tetrode','turns','magnitude','note','exception'],
                                   index=pd.DatetimeIndex, name='datetime')
            gsd.set_with_dataframe(self.raw_worksheet, self.df, include_index=True)

        # Check datetime index
        if self.df.index.name != 'datetime':
            self.df.set_index('datetime', inplace=True)

        # structure that tracks how time is being processed {manual|auto} ... automatic means
        # whenever data comes it, it receives the .utcnow() datetime. manual is when the
        # user tells the system how to time stamp all upcoming commmands. All upcoming commands
        # recevie that timestamp given until either a new time is given or the mode is switched to auto
        self.time = {'time':'', 'mode':'auto'}

    def backup(self, from_cloud=True):
        ''' Procedure for creating a rawbackup sheet '''
        if self.exist_worksheet("Raw Backup"):
            self.spreadsheet.del_worksheet(self.spreadsheet.worksheet('Raw Backup')) 
            self.spreadsheet.duplicate_sheet(self.raw_worksheet.id, new_sheet_name='Raw Backup') 
        else:
            if from_cloud:
                self.spreadsheet.duplicate_sheet(self.raw_worksheet.id, new_sheet_name='Raw Backup') 
            else:
                backup = self.spreadsheet.add_worksheet('Raw Backup', rows=1, cols=1)
                gsd.set_with_dataframe(backup, self.df, include_index=True)

    # GET METHODS
    @staticmethod
    def get_parameter(res, x):
        return res.get('queryResult').get('parameters').get(x)
    @staticmethod
    def get_parameters(res, X):
        return [res.get('queryResult').get('parameters').get(x) for x in X]

    @staticmethod
    def get_intent(res):
        '''
        Returns the intent of the conversation

        (For right now I have a hacky way of doing this)
        '''
        return res.get('queryResult').get('parameters').get('intent').lower()

    ## Validate worksheet
    # -------------------
    def validate_worksheet(self, sheet_name):
        ''' If sheet does not exist add it!  '''
        if not self.exist_worksheet(sheet_name):
            self.spreadsheet.add_worksheet(sheet_name, rows=1, cols=1)

    def exist_worksheet(self, sheet_name):
        '''Shortcut for checking sheet existence'''
        return sheet_name in [ws.title for ws in self.spreadsheet.worksheets()]

    ## ACCESSING DF INFO
    ## -------------
    def get_df_current_tetrode(self):
        ''' Returns current tetrode '''
        adjustDF        = self.df[self.df.intent == 'adjust-tetrode']
        current_tetrode = adjustDF.iloc[-1].tetrode
        return current_tetrode

    def get_datetime(self, mode=None):				
        ''' Return an index indicating the datetime '''
        if mode is None:
            mode = self.time['mode']
        if mode == "auto":
            I = pd.DatetimeIndex([arrow.now().datetime])
        elif mode == "manual":
            I = pd.DatetimeIndex([self.time['time']])
        else:
            raise ValueError('exerpimate time mode incorrect')
        I.name = 'datetime'
        return I

    ## GENERAL PARSING
    ## -------------
    def parse_magnitude(self, magnitude):
        ''' Converts parsing into pretty text '''
        if isinstance(magnitude, (tuple, list)):
            magnitude = ' '.join(magnitude)
        #magnitude = magnitude.replace('plus', '+')
        #magnitude = magnitude.replace('minus', '-')
        return magnitude

    ## Tetrode information fetching from google sheets
    ## -----------------------------------------------
    # Helps to process intelligent replies
    def fetch_tetrode_properties(self):
        ''' Fetch tetrode properties '''
        self.validate_worksheet('Mapping')
        self.tetrode_properties = \
            gsd.get_as_dataframe(self.spreadsheet.worksheet('Mapping'))
        (self.tetrode_properties
              .drop(columns=[x for x in self.tetrode_properties.columns 
                             if 'Unnamed' in x], 
                    inplace=True)
         )
        self.tetrode_properties.dropna(how='all', axis=1, inplace=True)
        self.tetrode_properties.dropna(how='any', axis=0, inplace=True)
        self.tetrode_properties.rename({x:str(x).lower().replace(' #', '') for x in
                                        self.tetrode_properties.columns}, 
                    inplace=True, axis=1)
        self.tetrode_properties.set_index('tetrode')

    def fetch_prior_stats(self, areas=['PFC','CA1']):
        ''' Fetch prior statistics about lowering '''
        import gspread
        self.validate_worksheet('Prediction')
        stats = {area:{} for area in areas}
        pworksheet = self.spreadsheet.worksheet("Prediction")
        stats['CA1']['median'] = float(pworksheet.acell('L2').value)
        stats['PFC']['median'] = float(pworksheet.acell('M2').value)
        stats['CA1']['lower']  = float(pworksheet.acell('L5').value)
        stats['PFC']['lower']  = float(pworksheet.acell('M5').value)
        stats['CA1']['upper']  = float(pworksheet.acell('L8').value)
        stats['PFC']['upper']  = float(pworksheet.acell('M8').value)
        self.prior_stats = stats

    ## Intelligent reply methods (get)
    # --------------------------------
    def get_depth(self, tetrode, depth_type='new', return_depth_data=False):
        ''' Returns a fulfillmentText about the depth, and mm '''
        if not isinstance(tetrode, list) and int(tetrode) == -1:
            tetrode = self.get_df_current_tetrode()
        if depth_type == "new":
            turns_per_mm = 4
        else:
            turns_per_mm = 1/0.3175
        
        depth = self.df[ self.df.tetrode == tetrode ].turns.fillna(0).cumsum().iloc[-1]
        markers_per_revolution = 12
        mmdepth = float(depth)/(markers_per_revolution * turns_per_mm)
        if return_depth_data:
            ret_value = depth, mmdepth
        else:
            pstring = self.prediction(tetrode)
            ret_value = f"Depth of {tetrode} is {depth:2.2f}. In other words, {mmdepth:2.2f} millimeters. {pstring}"
        return ret_value

    def prediction(self, tetrode, depth=None, return_prediction_data=False):
        '''
        Generates a prediction indicating how many turns remaining 
        Input
        ------
        tetrode : int
            which tetrode to estimate for
        Output
        ------
        if return_prediction_data: (2.5% percentile, median, 97.5% percentile)
        else: string(2.5% percentile, median, 97.5% percentile)
        '''

        try:
            # Do we have needed info from google sheet to estimate?
            if not hasattr(self, 'prior_stats') or self.prior_stats is None:
                self.fetch_prior_stats()
            if not hasattr(self, 'tetrode_properties') or self.tetrode_properties is None:
                self.fetch_tetrode_properties()

            # Lookup area of tetrode
            area = self.tetrode_properties[self.tetrode_properties.index == tetrode].area.iloc[0]
            # Get depth
            if depth is None:
                depth, _ = self.get_depth(tetrode, return_depth_data=True)
            # Get lower pecentile, median, and upper percentile
            areastat = self.prior_stats[area]
            # Find differences
            diff = {type_:(areastat[type_]*(0.3175)/(0.25)-depth) 
                    for type_ in ('lower','median','upper')}
            
            if return_prediction_data:
                ret_value = diff
            else:
                ret_value = f" <{diff['lower']:2.1f}, {diff['median']:2.1f}, {diff['upper']:2.1f}>"
        except Exception as E:
            ret_value = f""

        return ret_value

    def get_current_property_table(self, properties=['depth']):
        ''' Returns a table describing tetrodes and their current depths '''
        _, pretty_table = self.raw2pretty()

        current_depths = pretty_table.swaplevel(0,1,axis=1)[properties].ffill().iloc[-1]
        current_depths_df = pd.DataFrame(current_depths)
        current_depths_df.columns = list(properties)
        # Get area tetrodes
        current_depths_df = current_depths_df.merge(self.tetrode_properties, on="tetrode")
        current_depths_df.set_index(list(self.tetrode_properties.columns), append=True, inplace=True)
        return current_depths_df

    ## ENTRY METHODS
    ## -------------
    def entry_dead(self, tetrode, channel):
        raise NotImplementedError
    def entry_set_time(self, mode, time, add_to_df=True):
        if mode == "auto":
            self.time['mode'] = "auto"
            self.time['time'] = ''
        elif mode == "manual":
            self.time['mode'] = 'manual'
            D = arrow.get(time).datetime
            now = arrow.now().datetime
            # Since google assistant assumes if you don't give a year it's next year
            # I'm correcting that assumption here. A user of tetro-mate if they're
            # not giving a year means this year.
            if D.year > now.year:
                D = D.replace(year=now.year)
            self.time['time'] = D

        fulfillmentText = f"{self.time['mode']}: {arrow.get(self.time['time']).humanize()}, {arrow.get(self.time['time']).format('YYYY-MM-DD')}"

        if add_to_df:
            new_row = pd.DataFrame([['set-time', self.time['mode'], self.time['time']]], 
                               index=self.get_datetime(mode="auto"),
                               columns=['intent','mode','parameter'])
            self.df = pd.concat([self.df, new_row], axis=0)
        else:
            print(fulfillmentText)
        return fulfillmentText
    def entry_set_area(self, area, tetrode):
        if tetrode == -1:
            tetrode = self.get_df_current_tetrode()
        new_row = pd.DataFrame([['area', tetrode, area]], 
                               index=self.get_datetime(),
                               columns=['intent','tetrode','area'])
        if self.continuous_explode:
            new_row.explode('tetrode') # If a list of tetrodes given, explode the dataframe to proper entries
        self.df = pd.concat([self.df, new_row], axis=0)
    def entry_ripples(self, magnitude):
        magnitude = self.parse_magnitude(magnitude)
        new_row = pd.DataFrame([['ripple', self.get_df_current_tetrode(), magnitude]], 
                               index=self.get_datetime(),
                               columns=['intent','tetrode','magnitude'])
        self.df = pd.concat([self.df, new_row], axis=0)
        if self.continuous_explode:
            new_row = new_row.explode('tetrode')
        return "ripples = {} ✓".format(magnitude)

    def entry_theta(self, magnitude):
        magnitude = self.parse_magnitude(magnitude)
        new_row = pd.DataFrame([['theta', self.get_df_current_tetrode(), magnitude]], 
                               index=self.get_datetime(),
                               columns=['intent','tetrode','magnitude'])
        if self.continuous_explode:
            new_row = new_row.explode('tetrode')
        self.df = pd.concat([self.df, new_row], axis=0)
        return "theta = {} ✓".format(magnitude)

    def entry_delta(self, magnitude):
        magnitude = self.parse_magnitude(magnitude)
        new_row = pd.DataFrame([['delta', self.get_df_current_tetrode(), magnitude]], 
                               index=self.get_datetime(),
                               columns=['intent','tetrode','magnitude'])
        if self.continuous_explode:
            new_row = new_row.explode('tetrode')
        self.df = pd.concat([self.df, new_row], axis=0)
        return "delta = {} ✓".format(magnitude)

    def entry_cells(self, magnitude):
        magnitude = self.parse_magnitude(magnitude)
        new_row = pd.DataFrame([['cells', self.get_df_current_tetrode(), magnitude]], 
                               index=self.get_datetime(),
                               columns=['intent','tetrode','magnitude'])
        if self.continuous_explode:
            new_row = new_row.explode('tetrode')
        self.df = pd.concat([self.df, new_row], axis=0)
        return "cells = {} ✓".format(magnitude)

    def entry_adjust_tetrode(self, direction, tetrode, turns, append_prediction=True):

        if isinstance(direction, list):
            direction = ' '.join(direction)

        # Parse the direction and add any unrecognized to exception
        exception = ""
        if direction.lower().strip() in ('up', 'raise', 'rays', 'minus'):
            direction = -1
        elif direction.lower() in ('down','lower','plus'):
            direction = 1
        else:
            exception = "direction={}".format(direction)
        if len(exception) < 100:
            exception += " " * (100-len(exception))

        # Append new data to table
        signed_turn = float(turns) * float(direction)
        new_row = pd.DataFrame([['adjust-tetrode', tetrode, signed_turn, exception]], 
                               index=self.get_datetime(),
                               columns=['intent', 'tetrode', 'turns', 'exception'])
        if self.continuous_explode:
            new_row = new_row.explode('tetrode')
        self.df = pd.concat([self.df, new_row], axis=0)
        #self.df.loc[:, 'depth'] = self.df[self.df.intent == 'adjust-tetrode'].groupby('tetrode').turns.cumsum()

        if not isinstance(tetrode, list):
            depth, depthmm = self.get_depth(tetrode, 'new', return_depth_data=True)
            if append_prediction:
                pstring = self.prediction(tetrode)
            else:
                pstring = ''
        
            return f' ({depth:2.1f}, {depthmm:2.1f}) {pstring} ✓'
        else:
            tetrode = [str(tet) for tet in tetrode]
            return f" ✓"

    def entry_undo_entry(self):
        if len(self.df) > 0:
            self.df = self.df.iloc[:-1]
            return "Got it, deleting previous entry."
        else:
            return "Dataframe is empty"
    def entry_notes(self, note):
        ''' Notes at a given tetrode and adjustment '''
        new_row = pd.DataFrame([['notes', self.get_df_current_tetrode(), note]], 
                               index=self.get_datetime(),
                               columns=['intent','tetrode','note'])
        if self.continuous_explode:
            new_row = new_row.explode('tetrode')
        self.df = pd.concat([self.df, new_row], axis=0)
        return "note = {} ✓".format(note)

    def entry_marker(self, marker):
        ''' Markers to set time periods of lowering '''
        new_row = pd.DataFrame([['marker', marker]], 
                               index=self.get_datetime(),
                               columns=['intent','marker'])
        self.df = pd.concat([self.df, new_row], axis=0)
        return "marker = {} ✓".format(marker)

    # TYPE - CHECKING
    # ---------------
    def set_types(self, df=None, continuous_explode=None):
        '''
        Set types of our tracker dataframe that keeps tabs on all
        commands the user submits

        If df is provided, this works like a static function!
        '''
        import numpy as np
        if df is None:
            df = self.df
            modify_self = False # Modify the self obj or return at end?
        else:
            modify_self = True

        # Make index pd.Datetime!
        if "datetime" in df.columns:
            I = pd.to_datetime(df.datetime, utc=True)
            I = I.dt.tz_convert('EST')
            I = pd.DatetimeIndex(I)
            I.name = 'datetime'
            df = df.drop(columns='datetime').set_index(I)
        else:
            try:
                I = df.index
                #I = I.dt.tz_convert('EST')
                I = pd.DatetimeIndex(I)
                I.name = 'datetime'
                df = df.set_index(I)
            except Exception as E:
                try:
                    I = df.index
                    I = I.dt.tz_convert('EST')
                    I = pd.DatetimeIndex(I)
                    I.name = 'datetime'
                    df = df.set_index(I)
                except Exception as E:
                    I = df.index
                    I = I.tz_convert('EST')
                    I = pd.DatetimeIndex(I)
                    I.name = 'datetime'
                    df = df.set_index(I)

        # Determine tetrode type based on user settings
        if continuous_explode is None:
            continous_explode = self.continuous_explode
        if continuous_explode:
            tetrode_type = np.double
        else:
            tetrode_type = np.object

        # Retype
        df = df.astype({x:y for x,y in 
                                  {'tetrode':tetrode_type, 'turns':np.double, 'magnitude':np.double,
                               'adjustment':np.double, 'depth':np.double}.items()
                              if x in df.columns})

        if modify_self:
            self.df = df
        else:
            return df

    # EASIER TO READ TABLE GENERATION
    # -------------------------------
    def raw2pretty(self, from_cloud=False, subsummaries=True):
        ''' 
        Converts the table into the readable format we all know and love 
        '''
        import numpy as np

        # Determine data matrix
        if from_cloud:
            pass
        else:
            rawdf = self.df
        if rawdf.index.name != 'datetime':
            rawdf.set_index('datetime', inplace=True)
        # Explode any lists
        rawdf = rawdf.explode('tetrode')
        # Set types
        rawdf = self.set_types(rawdf, continuous_explode=False)

        # Preamble
        # --------
        grouping_name = 'day'
        grouping         = rawdf.index.day
        notation_intents = set(rawdf.columns) - {"adjust-tetrode", "marker"}

        ## UNIFIED ENTRY PER ADJUSTMENT ##
        ## -------------------------------
        # If markers are present, fill forward (for later!)
        markersPresent = 'marker' in rawdf.columns and not self.df.marker.isnull().all()
        # Add the current marker to each
        if markersPresent:
            I = rawdf.iloc[np.where(np.diff(self.df.index.day.values)>=1)[0]+1].index
            rawdf.loc[I,'marker'] = rawdf.loc[I,'marker'].fillna('')
            rawdf['marker'] = rawdf.marker.ffill()
        # Mark each note by the adjustment it belongs to
        rawdf['adjustment'] = rawdf.intent == 'adjust-tetrode'
        rawdf['adjustment'] = rawdf.adjustment.cumsum()
        def collapse_notes(frame):
            ''' 
            Collapses all notes per tetrode adjustment 
            '''
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

        ## ANNOTATE (including the grouping)
        ## ---------------------------------
        # Create the depth
        tetrodeAdjustments.loc[:,'depth']     = tetrodeAdjustments.groupby('tetrode').turns.cumsum()
        tetrodeAdjustments.loc[:,'depth_old'] = tetrodeAdjustments.loc[:,'depth'] * (0.25)/(0.3175)
        tetrodeAdjustments.loc[:,'depth_mm']  = (tetrodeAdjustments.loc[:,'depth']) / (4 * 12)
        predictions = [self.prediction(tetrode, tetrodeAdjustments.loc[tetrodeAdjustments.tetrode==tetrode,'depth']) 
                      for tetrode in tetrodeAdjustments.tetrode]
        # Add predictions
        #P = {}
        #for tetrode, prediction in enumerate(predictions):
        #    prediction = pd.DataFrame.from_dict(prediction)
        #    P[tetrode] = prediction
        #predictions = pd.DataFrame.from_dict(P, orient="index")
        #tetrodeAdjustments = pd.concat(tetrodeAdjustments, axis=0)
        ## Add remainingDepth
        #for _type in ('lower','median','upper'):
        #    tetrodeAdjustments.loc[:,'remainingDepth_' + _type] = tetrodeAdjustments.loc[:, _type] - tetrodeAdjustments.loc[:,'depth']

        # Label with grouping
        tetrodeAdjustments.loc[:, grouping_name] = tetrodeAdjustments.index.day
        # Determine the super group, which is the cartesian product of marker and grouping
        if markersPresent:
            tetrodeAdjustments.loc[:,'supergroup'] = (
                tetrodeAdjustments[grouping_name].astype('str') + ' - ' +
                tetrodeAdjustments['marker'].astype('str'))
            grouping = [grouping_name, 'marker']
        else:
            grouping = [grouping_name]
            #grouping = tetrodeAdjustments[grouping_name]

        ## PIVOT
        ## -----
        tetrodeAdjustments = (tetrodeAdjustments
         .drop(columns=[x for x in tetrodeAdjustments.columns if 'level_' in x])
         .reset_index()
         .drop(columns=[x for x in tetrodeAdjustments.columns if 'level_' in x])
          )
        pretty_table = []
        #grouping = np.unique(grouping, return_inverse=True)[1]
        for group, data in tetrodeAdjustments.groupby(grouping):
            print(f'Creating table for group = {group}')
            assert(not data.empty)
            assert((data.day == data.day.iloc[0]).all())
            # B. Reindex by the ordinal position within group per tetrode
            count_ordinal = lambda x : pd.Series(np.arange(len(x))+1, index=x.index)
            data = data.sort_values(['tetrode','datetime'])
            ordinals = pd.DataFrame(data.groupby('tetrode').apply(count_ordinal))
            ordinals.set_index(data.index, inplace=True)
            data['adjustment'] = ordinals
            
            #data = data.reset_index().set_index(['day','marker','ordinal'])
            # C. PIVOT
            pivot_values = ['turns','depth', 'depth_old', 'depth_mm', 'notes']
            data = (data
                    .fillna('')
                    .pivot_table(index=set(data.columns).intersection(['day','marker','adjustment']), 
                                 columns=['tetrode'],
                                 values=pivot_values,
                                 aggfunc=lambda X: "\n".join([str(x) for x in X]))
                    .swaplevel(0,1,axis=1)
                    .sort_index(axis=1)
                    )
            pretty_table.append(data)

        ## FINALIZE, STORE AND RETURN PRETTY TABLE
        ## ---------------------------------------
        pretty_table = pd.concat(pretty_table, axis=0)
        pretty_table_computable = pretty_table
        pretty_table = pretty_table.fillna('')
        self.df_pretty = pretty_table
        gsd.set_with_dataframe(self.pretty_worksheet,
                                  pretty_table.reset_index(), 
                                  include_index=False,
                                  include_column_header=True,
                                  allow_formulas=False)
        
        # Display short summaries)
        # ------------------------
        if subsummaries:
            for obj in pivot_values:
                worksheet = f"Summary_{obj}"
                self.validate_worksheet(worksheet)
                gsd.set_with_dataframe(self.spreadsheet.worksheet(worksheet),
                                       pretty_table.swaplevel(0,1, axis=1)[obj].reset_index(), 
                                       include_index=False,
                                       include_column_header=True,
                                       allow_formulas=False)
        # HTML Text
        # ---------
        #ExperimateLogger.html_summary(pretty_table_computable, filename='pretty.html')
        #ExperimateLogger.html_summary((pretty_table_computable
        #                               .swaplevel(0,1,axis=1)
        #                               .drop(columns='notes')
        #                               .swaplevel(0,1,axis=1)), 
        #                              filename='pretty_wonotes.html')

        return pretty_table, pretty_table_computable

    def pretty2raw(self, from_cloud=False):
        ''' Converts the table into the readable format we all know and love '''
        pass


    @staticmethod
    def html_summary(df, filename=None):

        def prepare_column(df, column):
            '''

            '''
            depth_locs           = df.swaplevel(0,1,axis=1).columns.get_loc(column)
            df.loc[:,depth_locs] = df.loc[:,depth_locs].astype('float')
            vmax                 = df.loc[:,depth_locs].values.ravel().max()
            vmin                 = df.loc[:,depth_locs].values.ravel().min()
            return depth_locs, vmax, vmin

        # Get ready to manipulate columns and format
        vmax = {}
        vmin = {}
        locs = {}
        for c, column in enumerate(('depth','remainingDepth')):
            locs[column], vmax[column], vmin[column] = prepare_column(df, column)
            if c == 0:
                overall_locs = locs[0]
            else:
                overall_locs = locs[0] | locs[c]

        print('Rendering')
        H = (df.style
               .bar(subset=locs['depth'], vmax=vmax['depth'])
               .bar(subset=locs['remainingDepth'], vmax=vmax['remainingDepth']-vmin['remainingDepth'])
               .highlight_null('grey')
               .set_caption('Tetrode Lowering Chart: Per day and day session (marker), each adjustment nth adjustment per tetrode. Tetrode and their properties are in the columns. The rows are the individual adjustment per session')
               .format(subset=overall_locs, formatter="{:20,.0f}")
               .render()
               .replace('nan','')
             )
        print('Done')
        if filename is not None:
            with open(filename, 'w') as F:
                F.write(H)
        return H

    def formatRaw(self):
        ''' Format raw '''
        import gspread_formatting as gsf
        # Format the header
        # -----------------
        fmt = cellFormat(
            backgroundColor=gsf.color(1, 0.9, 0.9),
            textFormat=textFormat(bold=True, foregroundColor=gsf.color(1, 0, 1)),
            )
        gsf.format_cell_range(worksheet, 'A1:MN1')

        # FORMAT THE SUBSEQUENT LINES
        # ---------------------------
        return None

    def formatPretty(self):
        ''' Format pretty '''
        import gspread_formatting as gsf
        # Format the header
        # -----------------
        fmt = cellFormat(
            backgroundColor=gsf.color(1, 0.9, 0.9),
            textFormat=textFormat(bold=True, 
                                  foregroundColor=gsf.color(1, 0, 1)),
            )
        gsf.format_cell_range(worksheet, 'A1:MN1')

        # FORMAT THE SUBSEQUENT LINES
        # ---------------------------
        return None

Elogger = ExperimateLogger

# WEB EXPOSED METHODS
@app.route("/")
def hello():
    return "Hello from APIAI Webhook Integration."

@app.route("/version")
def version():
    return "APIAI Webhook Integration. Version 1.0"

@app.route("/webhook", methods=['POST','GET'])
def webhook():

    import sys
    print('Received instruction', file=sys.stderr)

    # If update from cloud each time, then do it
    if EL.continuous_cloud_update:
        EL.df = gsd.get_as_dataframe(EL.raw_worksheet, include_index=True,
                                     parse_dates=True, userows=[0,1])

    # Obtain request object and parse JSON
    req = request.get_json(force=True)
    
    # Decode the intent
    intent = EL.get_intent(req)
    print(f'---------------', file=sys.stderr)
    print(f'Intent={intent}', file=sys.stderr)
    print(f'---------------', file=sys.stderr)

    # Default text to respond with unless modified
    fulfillmentText = req.get('queryResult').get('fulfillmentText')

    ## ADD TO TABLE
    intent_recognized = True
    if intent == "adjust-tetrode":
        # Acquire endogenous fulfillment text and return
        fulfillmentAddon = EL.entry_adjust_tetrode(
            *[Elogger.get_parameter(req, x) for x in
                                  ('direction','tetrode','turns')])
        fulfillmentText += fulfillmentAddon
    elif intent == "ripples":
        fulfillmentText  = EL.entry_ripples(Elogger.get_parameter(req, 'magnitude'))
    elif intent == "theta":
        fulfillmentText  = EL.entry_theta(Elogger.get_parameter(req,   'magnitude'))
    elif intent == "delta":
        fulfillmentText  = EL.entry_delta(Elogger.get_parameter(req,   'magnitude'))
    elif intent == "cells":
        fulfillmentText  = EL.entry_cells(Elogger.get_parameter(req,   'magnitude'))
    elif intent == "backup":
        fulfillmentText  = EL.backup()
    elif intent == "undo":
        fulfillmentText = EL.entry_undo_entry()
        # Acquire endogenous fulfillment text and return
        #fulfillmentText = req.get('queryResult').get('fulfillmentText')
    elif intent == "notes":
        fulfillmentText = EL.entry_notes(Elogger.get_parameter(req, 'note'))
    elif intent == "marker":
        fulfillmentText = EL.entry_marker(Elogger.get_parameter(req, 'marker'))
    elif intent == "raw2pretty":
        EL.raw2pretty()
    elif intent == "pretty2raw":
        EL.pretty2raw()
    elif intent == "change-tetrode":
        EL.entry_adjust_tetrode('down', Elogger.get_parameter(req, 'tetrode'), 0)
    elif intent == "get-depth":
        fulfillmentText = EL.get_depth(*Elogger.get_parameters(req, ['tetrode', 'depth_type']))
    elif intent == "set-area":
        fulfillmentText = EL.entry_set_area(*Elogger.get_parameters(req, ['area', 'tetrode']))
    elif intent == "set-time":
        fulfillmentText = EL.entry_set_time(*Elogger.get_parameters(req, ['mode', 'time']))
    elif intent == "dead":
        fulfillmentText = EL.entry_dead(*Elogger.get_parameters(req, ['tetrode', 'channel']))
    else:
        intent_recognized = False
        fulfillmentText = "Intent not recognized!"

    # Upload new table
    # ----------------
    gsd.set_with_dataframe(EL.raw_worksheet, EL.df, include_index=True, resize=True, allow_formulas=False)

    return jsonify({"fulfillmentText":fulfillmentText})

@app.route('/pretty', methods=['GET'])
def projects():
    #return app.send_static_file('/'.join((os.getcwd(),'pretty.html')))
    return app.send_static_file('pretty.html')
                                                                               
                                                                               
#  _____ _____ _____ _____ _____ _____ _____ _____ _____ _____ _____ _____ _____ 
# |_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|
#                                                                                
if __name__ == "__main__":

    # Access my Google Spreadsheet Credentials
    from oauth2client.service_account import ServiceAccountCredentials
    path = '/home/ryoung/GoogleAPI/myproj.json'
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(path, scope)
    url = 'https://docs.google.com/spreadsheets/d/1ho9GtnpfkY0KCpiy5HUSVQA0yqIL_-UyCWAUwJgY1jE/edit?usp=sharing'

    # Use that to setup an ExperimateLogger
    import experimate_webserver
    EL = experimate_webserver.ExperimateLogger(credentials=credentials, title='', key='', url=url, log_pretty_table=False)
    app.run()
