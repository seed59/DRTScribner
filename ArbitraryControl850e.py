import numpy as np
import csv
import math

#serves to write a file that the Scribner 850e uses for measurement procedures

class ArbitraryControl850e():
     def __init__(self, file_name):
        self.csvfile = open(file_name, 'w', newline='')
        arb_writer = csv.writer(self.csvfile, delimiter='\t',quotechar="'", quoting=csv.QUOTE_MINIMAL)
        self.writer = arb_writer

     def __del__(self):
        self.csvfile.close()

     def close_file(self):
        self.csvfile.close()

     def wait(self,wait_time):
          self.writer.writerow(['-1',str(wait_time)])

     def set_temperature(self, T_cell, T_anode, T_cathode):
          self.writer.writerow(['21',str(T_cell),str(T_anode),str(T_cathode),'0'])

     def measure_at_ocv(self, wait_time=10):
          self.writer.writerow(['2',str(-0.001),str(1)])
          self.writer.writerow(['0',str(wait_time)])

     def set_arbitrary_potential(self, potential):
          self.writer.writerow(['2',str(potential)])

     def measure_at_arbitrary_potential(self, potential, wait_time):
          self.set_arbitrary_potential(potential)
          self.measure_after_wait_time(wait_time)

     def set_potential_below_previous(self, potential_interval):
          if potential_interval>0:
            potential_interval=-potential_interval
          self.writer.writerow(['2',str(potential_interval),'2'])

     def measure_below_previous_potential(self,wait_time,potential_interval):
          self.set_potential_below_previous(potential_interval)
          self.measure_after_wait_time(wait_time)

     def set_arbitrary_current_density(self, current_density):
          self.writer.writerow(['11',str(current_density)])

     def set_current_density_above_previous(self, current_density):
          self.writer.writerow(['11',str(current_density),'1'])

     def measure_after_wait_time(self,wait_time):
          self.writer.writerow(['0',str(wait_time)])

     def generate_polarization(self, potential_interval, wait_time=10, number_of_points=None):
          if number_of_points==None:
            #we calculate the number of points as for a typical fuel cell polarization curve that runs from ca. 1V to ca. 0.35V
            number_of_points=round(0.65/potential_interval)
          if potential_interval>0:
            #for a FC polarization we will start close to OCV and go down in potential -- the potential steps should be negative
            potential_interval=-potential_interval

          #the first measurement point will  be at ocv (or very close to --- set 1mV below (-0.001))
          self.measure_at_ocv(wait_time=wait_time)

          for point in range(number_of_points):
               self.writer.writerow(['2',str(potential_interval),'2']) #the 2 at the end means to set the voltage below the previous voltage
               self.writer.writerow(['0',str(wait_time)])

     def generate_frequency_sweep(self,f_start,f_end,p_per_decade,amplitude=0.001,amplitude_in_percent=False,amplitude_minimum=0.001,amplitude_maximum=1):
          
          #get the exponents of the start and stop frequency
          flog_start = math.log(f_start, 10)
          flog_end = math.log(f_end, 10)
          
          #calculate the number of frequencies measured
          number_of_points = round((abs(flog_start)-abs(flog_end)))*p_per_decade

          #generate the frequency list on a log scale
          logspace_list = np.logspace(flog_start,flog_end,num=number_of_points)

          #lock the system in impedance mode for faster measurement
          self.writer.writerow(['42','1'])

          if amplitude_in_percent:
            for freq in logspace_list:
                 self.writer.writerow(['41',str(freq),str(amplitude_minimum),str(amplitude_maximum),str(amplitude)])
          else:
            for freq in logspace_list:
                 self.writer.writerow(['41',str(freq),str(amplitude)])

          #unlock the system from impedance mode
          self.writer.writerow(['42','0'])

     def turn_off_temperature(self):
          self.writer.writerow('30')

     def turn_off_fuel(self):
          self.writer.writerow('31')

     def turn_off_load(self):
          self.writer.writerow('32')
