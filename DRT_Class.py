import numpy as np
from scipy import integrate
import cvxpy as cp

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

"""
This class computes the distribution of relaxation times (DRT) from electrochemical impedance spectroscopy (EIS) data using Ridge regression.
The approach and problem discretization is implemented as in the publications: 
     Wan et al. Electrochemica Acta 2015, 184, 483-499
     Saccoccio et al. Electrochimica Acta 2014, 147, 470-482
and in their open source sotware DRTtools: 
     https://github.com/ciuccislab/pyDRTtools

"""
class DRT():
     def __init__(self, freq, Zre, Zim):
          #be careful to input here the Zim negative! and not -Zim, which is positive
          self.freq=freq
          self.taus=1./(2*np.pi*freq)
          self.taus_fine = np.logspace(np.log10(self.taus.min()),np.log10(self.taus.max()),5*freq.shape[0])
          self.N_freqs = freq.shape[0]
          self.N_taus = self.taus.shape[0]
          self.Z_exp_im = Zim
          self.Z_exp_re = Zre

     #setup the matrices for the discretized problem
     #pwl = piecewise linear
     #Gauss = Gaussian discretization (rbf = radial basis function)

     def compute_Aim_pwl(self):     
          #   compute number of frequency, tau and omega
          tau_vec = self.taus
          omega_vec = 2.*np.pi*self.freq

          #   define the A_im output matrix  #remember we sum over the taus
          out_A_im = np.zeros((N_freqs, N_taus))

          # use brute force
          for p in range(0, self.N_freqs):
            for q in range(0, self.N_taus):   
               if q == 0:
                   out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q+1]/tau_vec[q])
               elif q == N_taus-1:
                   out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q]/tau_vec[q-1])
               else:
                   out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*np.log(tau_vec[q+1]/tau_vec[q-1])                    
          return out_A_im

     def compute_A_Gauss(self, whichA='im'):
          tau_vec = self.taus
          omega = 2.*np.pi*self.freq
          out_A = np.zeros((self.N_freqs, self.N_taus))
          for p in range(0, self.N_freqs):
                 for q in range(0, self.N_taus):
                    if q != self.N_taus-1:
                         delta_log_tau = np.log(tau_vec[q+1])-np.log(tau_vec[q])
                    else:
                         delta_log_tau = np.log(tau_vec[q])-np.log(tau_vec[q-1])
                    epsilon = 0.5*(2.3548**2/(4*(delta_log_tau)**2))  #this matches the FWHM of the Gaussian to the difference between the taus
                    alpha = omega[p]*tau_vec[q]
                    if whichA=='im':
                         integrant = lambda x: alpha/(1./np.exp(x)+(alpha**2)*np.exp(x))*np.exp(-epsilon*(x**2))
                         out_A[p, q] = -integrate.quad(integrant, -20, 20, epsabs=1E-9, epsrel=1E-9)[0]
                    elif whichA=='re':
                         integrant = lambda x: 1./(1.+(alpha**2)*np.exp(2.*x))*np.exp(-epsilon*(x**2))
                         out_A[p, q] = integrate.quad(integrant, -20, 20, epsabs=1E-9, epsrel=1E-9)[0]
          return out_A

     #functions for using cvxpy - implemented as in their online example for Ridge regression
     #we cannot use sklearn because their implementation to keep the coefficients greater than zero does not work well
     #the coefficients are resistances and need to be necessarily greater than zero
     #beta are the coefficient we are searching for, alpha is the regularization parameter
     def loss_function(self, A, Z_exp, beta):
          return cp.norm(A@beta-Z_exp,p=2)**2

     def regularizer(self, beta):
          return cp.norm(beta,p=2)**2

     def objective_fct(self, A, Z_exp, beta, alpha):
          return self.loss_function(A,Z_exp,beta)+alpha*self.regularizer(beta)

     #mse = mean square error
     def mse(self, A, Z_exp, beta):
          t_size = A.shape[1] #t_size is either N_taus for the imaginary part or N_taus+1 for the real part where we also fit the intercept
          return (1.0/t_size)*self.loss_function(A,Z_exp,beta).value

     def fitRidge(self, alpha, whichA='im'):  
          if whichA=='im':
               self.A_im = self.compute_A_Gauss(whichA='im')
               h = np.zeros(self.N_taus)
               beta = cp.Variable(self.N_taus, value = np.ones(self.N_taus))
               #setup the problem
               problem = cp.Problem(cp.Minimize(self.objective_fct(self.A_im,self.Z_exp_im,beta,alpha)), [beta >= h])
               #solve and output the error
               problem.solve()
               print(f'error for imaginary part {self.mse(self.A_im,self.Z_exp_im,beta.value)}')
               #xs are betas - names used iterchangably (paper or program name versions are different)
               self.x_im=beta.value
               
          elif whichA=='re':
               A_re = self.compute_A_Gauss(whichA='re')
               #add a column to compute the intercept
               A = np.zeros((self.N_freqs, self.N_taus+1))
               A[:,1:] = A_re
               A[:,0] = 1
               self.A_re = A.copy()
               t_size = A.shape[1]
               h = np.zeros(t_size)
               #don't penalize the intercept
               beta_values = np.ones(t_size)
               beta_values[0] = 0
               #setup the problem
               beta = cp.Variable(t_size, value = beta_values)
               problem = cp.Problem(cp.Minimize(self.objective_fct(self.A_re,self.Z_exp_re,beta,alpha)), [beta >= h])
               #solve and output the error
               problem.solve()
               print(f'error for real part {self.mse(self.A_re,self.Z_exp_re,beta.value)}')
               self.x_re=beta.value 

          #output the coefficients
          return beta.value

     # convert the coefficients x to the DRT (gamma)
     def x2gamma(self, x):
          #xs are just the coefficients of the discretized gamma - we need to still multiply them by the RBF (the B matrix)
          tau_vec = self.taus
          N_tau_map = self.taus_fine.shape[0]
          B = np.zeros([N_tau_map, self.N_taus])
          for p in range(0, N_tau_map):
                 for q in range(0, self.N_taus):
                    if q != self.N_taus-1:
                         delta_log_tau = np.log(tau_vec[q+1])-np.log(tau_vec[q])
                    else:
                         delta_log_tau = np.log(tau_vec[q])-np.log(tau_vec[q-1])
                    epsilon = 0.5*(2.3548**2/(4*(delta_log_tau)**2))
                    delta_log_tau_map = np.log(self.taus_fine[p])-np.log(tau_vec[q])
                    B[p,q] = np.exp(-epsilon*(delta_log_tau_map**2))
          
          gamma = B@x
          return gamma

     def fit2gamma(self, alpha, whichA='im'):
          if whichA=='im':
               xs=self.fitRidge(alpha)
               #get the model impedance
               self.Z_model_im = self.A_im@xs
               #compute Z_model_re with the A_im matrix for later error analysis
               #need to get A_re without the intercept dimension addition
               A_re = self.compute_A_Gauss(whichA='re')     
               self.Z_model_re = A_re@xs

               self.gamma_im= self.x2gamma(xs)
               return self.gamma_im, self.Z_model_re, self.Z_model_im

          elif whichA=='re':
               xs=self.fitRidge(alpha,whichA='re')
               #get the model impedance
               self.Z_model_im = self.A_im@xs[1:]
               self.Z_model_re = self.A_re@xs

               #to get the DRT we remove the intercept
               self.gamma_re= self.x2gamma(xs[1:])
               return self.gamma_re, self.Z_model_re, self.Z_model_im

     def optimizeAlpha(self, alphas_vals):
          #calculate both cross validation and discrepancy to optimize the value of the regularization parameter alpha

          #define regularization parameter alpha
          alpha = cp.Parameter(nonneg=True)

          #get A matrices
          A_re = self.compute_A_Gauss(whichA='re')
          A_im = self.compute_A_Gauss(whichA='im')
          #
          #setup the problem
          #
          beta_im = cp.Variable(self.N_taus, value = np.ones(self.N_taus))
          beta_re = cp.Variable(self.N_taus, value = np.ones(self.N_taus))
          #imaginary problem
          problem_im = cp.Problem(cp.Minimize(self.objective_fct(A_im,self.Z_exp_im,beta_im,alpha)), [beta_im >= np.zeros(self.N_taus)])
          #real problem
          problem_re = cp.Problem(cp.Minimize(self.objective_fct(A_re,self.Z_exp_re,beta_re,alpha)), [beta_re >= np.zeros(self.N_taus)])

          coeffs_values_im = []
          coeffs_values_re = []

          #save also the mses
          errors_im = []
          errors_re = []

          discrepancy = []
          cross_validation = []
          #solve the problems for different values of alpha
          for val in alphas_vals:
              alpha.value = val
              
              problem_im.solve()
              mse = self.mse(A_im,self.Z_exp_im,beta_im.value)
              print(f'error for imaginary part {mse}')
              errors_im.append(mse)
              coeffs_values_im.append(beta_im.value)
              ZCVr = A_re@beta_im.value

              problem_re.solve()
              mse = self.mse(A_re,self.Z_exp_re,beta_re.value)
              print(f'error for real part {mse}')
              errors_re.append(mse)
              coeffs_values_re.append(beta_re.value)
              ZCVi = A_im@beta_re.value

              discrepancy.append(np.linalg.norm(beta_im.value-beta_re.value))
              cross_validation.append(np.linalg.norm(self.Z_exp_im-ZCVi)+np.linalg.norm(self.Z_exp_re-ZCVr))
          return discrepancy, cross_validation, coeffs_values_im, coeffs_values_re, errors_im, errors_re

     ####
     #from here on methods that do not call fitRidge anymore - the computationally less intensive part
     ####

     def plot_CoeffsAlpha(self, alphas, coeffs, ax=None, **plt_kwargs):
          if ax==None:
               ax=plt.gca()
          for i in range(len(alphas)):
               ax.semilogx(alphas, [xi[i] for xi in coeffs],**plt_kwargs)
          ax.set_xlabel(r'$\alpha$')
          ax.set_ylabel(r'$x_{i}$')
          return ax
          
     def plot_DiscrepancyCV(self, alphas, discrepancy, cross_validation, error=None, ax=None, **plt_kwargs):
          if ax==None:
               ax=plt.gca()
          if isinstance(ax,list):
               ax1=ax[0]
               ax2=ax[1]
          else:
               ax1=ax
               ax2=ax.twinx()

          ax1.loglog(alphas,cross_validation,'-xr',label='CV')
          ax1.set_ylabel(r'Cross Validation Error \ $\Omega^{2}$')
          #change the colors of the first axes (CV) to red
          ax1.yaxis.label.set_color('r')

          ax2.loglog(alphas,discrepancy,'-ok',label='discrepancy')
          ax2.set_ylabel(r'Discrepancy Error \ $\Omega^{2}$')

          ax1.set_xlabel(r'alpha')

          if error != None:
               ax3=ax1.twinx()
               ax3.loglog(alphas,error,'-^b',label='MSE')
               ax3.set_ylabel(r'Mean Square Error \ $\Omega^{2}$')
               #aesthetics for the axis:
               ax3.yaxis.label.set_color('b')
               ax3.spines["right"].set_color('b')
               ax3.spines['left'].set_color('r')
               ax3.spines["right"].set_position(("axes", 1.35))
          
          ax1.grid(which='both',linestyle = '--', linewidth = 0.2)
          #aesthetics of tick label formatting
          ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())
          ax2.yaxis.set_minor_formatter(mticker.ScalarFormatter())
          plt.tight_layout()
          return [ax1,ax2]

     def get_gamma(self,whichA='im'):
          if whichA=='im':
               return self.gamma_im
          elif whichA=='re':
               return self.gamma_re

     def get_Zmodel(self,whichA='im'):
          if whichA=='im':
               return self.Z_model_im
          elif whichA=='re':
               return self.Z_model_re

     def get_peaksHeightPosition(self,gamma_in):
          #find local max
          c = (np.diff(np.sign(np.diff(gamma_in))) < 0).nonzero()[0] + 1 # local max
          return gamma_in[c], self.taus_fine[c]
