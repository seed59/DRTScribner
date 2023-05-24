import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

from DataFile import *
from DRT_Class import *

########################################
#globally set the figure parameters
params = {'font.family': 'Arial',
          'legend.fontsize': 'medium',
          'figure.figsize': (7, 4),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'large',
          'axes.linewidth': '1.7',
          'xtick.major.width': '1.7',
          'ytick.major.width': '1.7',
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'axes.formatter.limits': (-3, 3),
          'axes.formatter.use_mathtext': False,
          'axes.formatter.useoffset': True,
          'axes.formatter.offset_threshold': 3}
pylab.rcParams.update(params)


######################################
##     Data import + some plots
######################################
DF = DataFile('measurement_data.txt')
#parse the data
all_DC, all_AC = DF.parse_FromARBfile('arbitrary_measurement_procedure.txt')

#plot max fuel cell power at different dew points
fig0,ax0 = plt.subplots(dpi=150)
DF.plot_TempOptimization(all_DC,ikey='Power_density',func='max',invert_colormap=False,ax=ax0)

#plot the polarizations
all_pols = DF.parse_Temperatures(all_DC)
fig1,ax1 = plt.subplots(dpi=150)
DF.plot_AllPols(all_pols,ax=ax1)

######################################
##         DRT demonstration
######################################
#for demonstrating the DRT analysis get a single 'random' EIS spectrum at a cathode and anode temperatures of 78, 78
single_eis,_ = DF.get_EisSingle(all_AC,[78,78],3)

#create DRT object from the EIS spectrum
freq = single_eis['Z_Freq'].to_numpy()
Zim = single_eis['Z_Imag'].to_numpy()
Zre = single_eis['Z_Real'].to_numpy()
#create DRT object
meaDRT = DRT(freq,Zre,Zim)

#REGRESSION - GET DRT
gamma_i, Zmir, Zmii = meaDRT.fit2gamma(alpha=0.001,whichA='im')
gamma_r, Zmrr, Zmri = meaDRT.fit2gamma(alpha=0.001,whichA='re')

#PLOT DRT
fig2,ax2 = plt.subplots(dpi=150)
ax2.semilogx(1./(2*np.pi*meaDRT.taus_fine), gamma_i*DF.surface_area)
ax2.set_ylabel(r'$\gamma (ln (\tau)) / \Omega  cm^{2}$')
ax2.set_xlabel(r'$f$ / Hz')
ax2.grid(visible=True, which='both',axis="both",linestyle = '--', linewidth = 0.2)


#PLOT EXPERIMENTAL AND MODELED EIS
fig3,ax3 = plt.subplots(dpi=150)
#experimental
ax3.plot(Zre*DF.surface_area,-Zim*DF.surface_area,'x-k',label='Experiment')
#model
ax3.plot(Zmrr*DF.surface_area,-Zmri*DF.surface_area,'-r',label='Model')
ax3.legend()
ax3.set_xlabel(r'$Z_{re} / \Omega cm^{2}$')
ax3.set_ylabel(r'$-Z_{im} / \Omega cm^{2}$')
#set equal aspect ratio
ax3.set_aspect('equal', 'box')
ax3.grid(b=True, which='both',axis="both",linestyle = '--', linewidth = 0.2)


#PLOT DISCREPANCY AND CROSS-VALIDATION
alphas = np.logspace(-5,1,20)
discrepancy, cross_validation, coeffs_values_im, coeffs_values_re, error_im, error_re = meaDRT.optimizeAlpha(alphas)
fig4,ax4 = plt.subplots(figsize=(10,4))
error_sum = [a + b for a, b in zip(error_im,error_re)]
meaDRT.plot_DiscrepancyCV(alphas,discrepancy, cross_validation,error=error_sum,ax=ax4)


plt.show()