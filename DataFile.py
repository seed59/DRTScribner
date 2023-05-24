import pandas as pd
import numpy as np
import copy
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt

from si_prefix import si_format
import collections

"""
DataFile class imports the measurement data from the Scribner machine and parses it
INPUT: infile should hold all the data in the format of the Scribner output
ATTRIBUTES: df dataframe with all the stored measurement points, a units dictionary, where the keys are the column names of the dataframe
  and a surface_area output, that can be used for varius normalizations
  (the units are important, e.g. depending on the current range used Scribner can choose mA/cm2 or A/cm2)

when the Scribner arbitrary measurement set up is used:
      - the usual  measurement is set up to measure the fuel cell voltage at a set current density, after which an impedance spectrum is taken
      - this procedure is repeated at various cathode and anode dew points (relative humidities)
      - the file infile holds all the data and the file ArbFile (passed to the parse_FromARBfile() function) 
      should hold the concreate measurement procedure as described in the Scribner manual

      this class parses all the data:
          temperature
          dc measurements
          ac measurements
"""


class DataFile():

   def __init__(self, infile):
      self.infile = infile
      self.df, self.units = self.import_File()
      self.surface_area = self.get_SurfaceArea()

   def get_ExperimentDescription(self):
      with open(self.infile) as f:
         for num,line in enumerate(f):
             if "Begin Experiment:" in line:
                 return line.split("Experiment:",1)[1]

   def get_SurfaceArea(self):
      with open(self.infile) as f:
         for num,line in enumerate(f):
             if "Surface" in line:
                 line=line.split()
                 return float(line[1])

   def find_Header(self):
      with open(self.infile) as f:
         for num,line in enumerate(f):
             if line.lstrip().startswith("Time ("):
                 return num

   def import_File(self):
      data = pd.read_table(self.infile, header=self.find_Header())
      df = pd.DataFrame(data)
      #sometimes the comments get imported as a extra row of NaNs, which makes problems
      df = df[df.iloc[:, 0] != 'End Comments']
      
      #separate the units
      units = {}
      columnnames = []
      for col in df:
         name, unit = col.split('(')
         #remove leading and trailing white spaces
         name = name.strip()
         unit = unit.strip()
         #some column names are used twice - rename them to have unique values
         if name == 'I' and 'cm' in unit[:-1]:
               name = 'I_density'
         if name == 'E_iR_Stack' and 'Ohm' in unit[:-1]:
               name = 'R_iR_Stack'
         if name == 'Power' and 'cm' in unit[:-1]:
               name = 'Power_density'
         if name == 'Flow_Anode' and 'stoich' in unit[:-1]:
               name = 'Anode_stoich'
         if name == 'Flow_Cathode' and 'stoich' in unit[:-1]:
               name = 'Cathode_stoich'

         #remove also the second bracket of the units with [:-1]
         units[name] = unit[:-1]
         columnnames.append(name)
      #change the column names to ones without units
      df.columns = columnnames

      #add some missing columns
      if 'Power_density' not in df.columns:
         surfacearea=self.get_SurfaceArea()
         df['Power_density'] = df['Power'].divide(surfacearea)
         units['Power_density'] = units['Power']+'/'+'cm2'
  
      return df, units

   #all the methods below deal with the arbitrary Scribner format
   #this means measurements at variout dew points, impedance and polarization

   def is_Arbitrary(self):
      print(self.get_ExperimentDescription())
      if 'Arbitrary' in self.get_ExperimentDescription():
        return True
      else:
        return False

   def parse_FromARBfile(self,ArbFile):
      #this function will take as a template the ArbFile that we used to run the Scribner and output two dataframes:
        # df_dc -- these are the usual polarization measurements
        # df_ac -- these are the EIS (electrochemical impedance spectr.) measurements
      arb_file = pd.read_table(ArbFile, sep='\t', header=0)
      arb_file_data = np.squeeze(arb_file.to_numpy())

      occurrences_imp = np.where(arb_file_data[:,0] == 42)  #42 is a unique definition of the Scribner machine to measure impedance
      self.length_imp = occurrences_imp[0][1]-occurrences_imp[0][0] -1
      
      arb_file_mask = np.logical_or(arb_file_data[:,0] == 0, arb_file_data[:,0] == 41)
      arb_file_data = arb_file_data[arb_file_mask,:]

      df_dc = self.df[arb_file_data[:,0]==0]
      df_ac = self.df[arb_file_data[:,0]==41]

      print(f'Imported form Arbitrary file.')
      print(f'Size of all impedance measurement data points is {df_ac.shape}.')
      return df_dc, df_ac

   def get_PolSingle(self,df_dc,temp):
      #output a single polarization curve at a specific anode and cathode dew point passed as a lit temp, e.g. [78,78]
      df_dc_Temp_mask = (df_dc['Ctrl_Temp_Anode']==temp[0]) & (df_dc['Ctrl_Temp_Cathode']==temp[1])
      return df_dc[df_dc_Temp_mask]

   def get_EisSingle(self,df_ac,temp,pot_idx):
      #output a single EIS spectrum at a specific anode and cathode dew point (temp, e.g. [78,78]), and a specific current density (pot_idx)
      #pot_idx is passed as an integer, because the specific current densities we measured at are not known
      
      df_ac_Temp_mask = (df_ac['Ctrl_Temp_Anode']==temp[0]) & (df_ac['Ctrl_Temp_Cathode']==temp[1])
      df_ac_Temp = df_ac[df_ac_Temp_mask]

      #parse the potentials
      df_grouped = self.parse_SingleDFAC(df_ac_Temp)
      eis_single = df_grouped.get_group(pot_idx)
      I_pot = [round(eis_single['I_density'].mean(),2),round(eis_single['E_Stack'].mean(),2)]
      return df_grouped.get_group(pot_idx), I_pot

   def build_ExperimentSummaryAC(self, df_ac, first_key='Temp'):
      df_ac_grouped = self.parse_Temperatures(df_ac)
      l_out = []
      d_out = collections.defaultdict(dict)

      #loop through temperatures
      for _,dfa0 in df_ac_grouped:
        dfa_grouped=self.parse_SingleDFAC(dfa0)
        exp_l = []
        pot_list = []
        if first_key=='Temp':
          Ta = dfa0['Ctrl_Temp_Anode'].iloc[0]
          Tc = dfa0['Ctrl_Temp_Cathode'].iloc[0]
          exp_l.append([Ta,Tc])
        elif first_key=='RH':
          RHc = dfa0['RH_Cathode'].mean() 
          RHa = dfa0['RH_Anode'].mean()
          exp_l.append([RHa,RHc])
        #loop through EIS at different current densities
        for _,dfa1 in dfa_grouped:
          if len(dfa1)>1:
            pot = round(dfa1['E_Stack'].mean(),2)
            cur_density = round(dfa1['I_density'].mean())
            pot_list.append([pot,cur_density])
            if first_key=='Temp':
              d_out[tuple([Ta,Tc])][tuple([pot,cur_density])] = None
            elif first_key=='RH':
              d_out[tuple([RHa,RHc])][tuple([pot,cur_density])] = None
        exp_l.append(pot_list)
        l_out.append(exp_l)
      return d_out,l_out

   def build_ExperimentSummaryDC(self, df_dc):
      df_dc_grouped = self.parse_Temperatures(df_dc)
      d_out={}
      #loop through temperatures
      for _,dfa0 in df_dc_grouped:
        Ta = dfa0['Ctrl_Temp_Anode'].iloc[0]
        Tc = dfa0['Ctrl_Temp_Cathode'].iloc[0]
        d_out[(Ta,Tc)] = None
      return d_out

   def parse_Temperatures(self,df):
      anode_shift = (df['Ctrl_Temp_Anode'].shift() != df['Ctrl_Temp_Anode']).cumsum()
      cathode_shift = (df['Ctrl_Temp_Cathode'].shift() != df['Ctrl_Temp_Cathode']).cumsum()
      return df.groupby([anode_shift, cathode_shift])

   def parse_SingleDFAC(self,df):
      #this should divide several impedance spectra into a single one
      df_grouped=df.groupby(np.arange(len(df)) // self.length_imp)
      return df_grouped

   def build_ARBMatrix(self, df, ikey='Power_density',func='max'):
      #builds a matrix for 3d plotting with a quantity determined by ikey that will be maximized (or mean or min depending on func) for each run
      #get unique anode and cathode temperatures
      Ts_anode = np.sort(df['Ctrl_Temp_Anode'].unique())
      Ts_cathode = np.sort(df['Ctrl_Temp_Cathode'].unique())

      #get all the polarization curves at different anode and cathode temps
      df_grouped=self.parse_Temperatures(df)

      #initilize the power matrix
      power_matrix = np.zeros([Ts_cathode.size,Ts_anode.size])

      #loop through all the polarization curves and get the maximum power
      for idx, gr in df_grouped:
        if func=='max' or func=='Max' or func=='MAX':
          power = gr[ikey].max()
        elif func=='min' or func=='Min' or func=='MIN':
          power = gr[ikey].min()
        elif func=='mean' or func=='Mean' or func=='MEAN':
          power = gr[ikey].mean()
        anode_index = np.where(Ts_anode==gr['Ctrl_Temp_Anode'].iloc[0])
        cathode_index = np.where(Ts_cathode==gr['Ctrl_Temp_Cathode'].iloc[0])
        #rows are y axis = cathode, columns are x axis = anode
        power_matrix[cathode_index[0],anode_index[0]] = power
        #output matrix, xticks, yticks
      return power_matrix, Ts_anode, Ts_cathode

   def get_BestGroup(self,df):
      #returns the polarization curve with the highest power density
      df_grouped = self.parse_Temperatures(df)
      #get the maximum powers in each group
      gr_max = df_grouped['Power_density'].max()
      #get the maximum of maximum
      gr_max_overall = gr_max.max()
      #get which group this belongs to
      for idx, gr in df_grouped:
        if gr['Power_density'].max() == gr_max_overall:
          return gr.reset_index()

   def plot_TempOptimization(self, df, ikey='Power_density',func='max',tick_labels='Temp', invert_colormap=False, ax=None, **plt_kwargs):
      #plots the power density (or other quantity depending on ikey) of the fuel cells as a function of cathode and anode dew points
      if ax==None:
          ax=plt.gca()

      power_matrix,xTs,yTs =self.build_ARBMatrix(df, ikey=ikey,func=func)
      #replace zeroes with nans
      power_matrix[power_matrix == 0] = np.nan
      #cmap
      if invert_colormap:
        cmap=copy.copy(cm.plasma_r)
      else:
        cmap=copy.copy(cm.plasma)
      #set nans to white -- not measured -- not in colormap
      cmap.set_bad('white')

      
      if tick_labels == 'Temp':
        im=ax.matshow(power_matrix, interpolation='none',cmap=cmap, extent=[np.min(xTs),np.max(xTs),np.max(yTs),np.min(yTs)],**plt_kwargs)

        #ticks
        ax.set_xticks(xTs)
        ax.set_yticks(yTs[::-1])
        #axis labels
        ax.set_xlabel('Anode Temperature' + ' / ' + self.units['Ctrl_Temp_Anode'])
        ax.set_ylabel('Cathode Temperature' + ' / ' + self.units['Ctrl_Temp_Cathode'])

      elif tick_labels == 'RH':
        #dew points set the relative humidity (RH) - is easier to understand - use this axis in papers/presentations
        RH_anode = []
        RH_cathode = []

        for tx in xTs:
          df_Ta = df[df['Ctrl_Temp_Anode'] == tx]
          RH_anode.append(round(df_Ta['RH_Anode'].mean()))
        for ty in yTs:
          df_Tc = df[df['Ctrl_Temp_Cathode'] == ty]          
          RH_cathode.append(round(df_Tc['RH_Cathode'].mean()))
        im=ax.matshow(power_matrix, interpolation='none',cmap=cmap, extent=[np.min(RH_anode),np.max(RH_anode),np.max(RH_cathode),np.min(RH_cathode)],**plt_kwargs)


        #ticks
        ax.set_xticks(RH_anode)
        ax.set_yticks(RH_cathode[::-1])
        #axis labels
        ax.set_xlabel('Anode Humidity' + ' / ' + '%')
        ax.set_ylabel('Cathode Humidity' + ' / ' + '%')

      #colorbar
      cbar=plt.colorbar(im)
      cbar.set_label(func +' '+ ikey + ' / ' + self.units[ikey])
      #set title with cell temperature (the cell temperature is what set the RH - needs to be mentioned on the graph)
      ax.set_title('Cell Temp.' + str(round(df['Temp'].iloc[0])) + ' / ' + self.units['Temp'])

   def get_DCColors(self,df_Tparsed):
      #helper function to plot the polarization in the same color scheme as the function plot_TempOptimization
      #df_Tparsed is the output of self.parse_Temperature
      #get a ordered list of colors that one can use to plot the polarizations
      #the colors correspond to the colormap that was plotted by the plot_TempOptimization function.
      num_measuerments = df_Tparsed.ngroups
      cmap = copy.copy(cm.plasma)

      power = []
      for idx, gr in df_Tparsed:
        power.append(gr['Power_density'].max())
      #convert power to values between 0 and 1
      power = [(i - min(power)) / (max(power)-min(power)) for i in power]
      colors=[]
      for nm in power:
        colors.append(cmap(nm))
      return colors

   def plot_AllPols(self,df_Tparsed,ax=None,**plt_kwargs):
      if ax==None:
            ax=plt.gca()

      num_measurements = df_Tparsed.ngroups
      if num_measurements > 1:
        colors = self.get_DCColors(df_Tparsed)
        cmap=mpl_colors.LinearSegmentedColormap.from_list('pol_power', colors, N=num_measurements)
        #arrange for colorbar
        c = np.arange(1, num_measurements+1)
        # Make dummie mappable
        dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)
        # Clear axis
        ax.cla()
      else:
        colors = [[0, 0, 0]]
      #setup list for labels
      cm_labels=[]
      
      
      ic=0
      for _, gr in df_Tparsed:
        cm_label = r'T$_{A}$='+str(gr['Ctrl_Temp_Anode'].iloc[0])+u'\u00b0'+ 'C'+'\n'+r'T$_{C}$='+str(gr['Ctrl_Temp_Cathode'].iloc[0])+u'\u00b0'+ 'C'
        cm_labels.append(cm_label)
        ax.plot(gr['I_density'],gr['E_Stack'],marker='o',color=colors[ic],label=cm_label,**plt_kwargs)
        ic+=1
      #axis labels
      ax.set_xlabel('Current' + ' / ' + self.units['I_density'])
      ax.set_ylabel('Voltage' + ' / ' + self.units['E_Stack'])
      if num_measurements > 1:
        #make colorbar
        fig = plt.gcf()
        cbar=fig.colorbar(dummie_cax, ticks=c)
        cbar.ax.set_yticklabels(cm_labels,fontsize=11)
        cbar.ax.get_yaxis().labelpad = 10
      plt.grid(which='both',linestyle = '--', linewidth = 0.2)
      return ax