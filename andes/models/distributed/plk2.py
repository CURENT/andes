"""
DER protection model.
"""
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.model import Model, ModelData
from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import ConstService
from andes.core.discrete import Limiter, Delay
 
 
class PLK2Data(ModelData):
   """
   PLK2 model data.
   """
 
   def __init__(self):
       super(PLK2Data, self).__init__()
       self.dev = IdxParam(info='idx of the target device',
                           mandatory=True,
                           )
       self.busfreq = IdxParam(model='BusFreq',
                               info='Target device interface bus measurement device idx',
                               )
 
       # -- protection enable parameters
       self.fena = NumParam(default=1,
                            tex_name='fena',
                            vrange=[0, 1],
                            info='Frequency deviation protection enable. \
                                  1 for enable, 0 for disable.',
                            )
       self.Vena = NumParam(default=0,
                            tex_name='Vena',
                            vrange=[0, 1],
                            info='Voltage deviation protection enable.\
                                  1 for enable, 0 for disable.',
                            )
 
       # -- protection parameters, frequency
       self.fl1 = NumParam(default=59.2,
                           tex_name='fl1',
                           info='Under frequency shadding point 1',
                           unit='Hz',
                           )
       self.fu1 = NumParam(default=60.5,
                           tex_name='fu1',
                           info='Over frequency shadding point 1',
                           unit='Hz',
                           )
       self.fl2 = NumParam(default=57.5,
                           tex_name='fl2',
                           info='Over frequency shadding point 2',
                           unit='Hz',
                           )
       self.fu2 = NumParam(default=61.5,
                           tex_name='fu2',
                           info='Over frequency shadding point 2',
                           unit='Hz',
                           )
       self.fl3 = NumParam(default=50,
                           tex_name='fl3',
                           info='Under frequency shadding point 3',
                           unit='Hz',
                           )
       self.fu3 = NumParam(default=70,
                           tex_name='fu3',
                           info='Over frequency shadding point 3',
                           unit='Hz',
                           )
 
       self.Tfl1 = NumParam(default=300,
                            tex_name=r't_{fl1}',
                            info='Stand time for under frequency 1',
                            non_negative=True,
                            )
       self.Tfl2 = NumParam(default=10,
                            tex_name=r't_{fl2}',
                            info='Stand time for under frequency 2',
                            non_negative=True,
                            )
       self.Tfu1 = NumParam(default=300,
                            tex_name=r't_{tu1}',
                            info='Stand time for over frequency 1',
                            non_negative=True,
                            )
       self.Tfu2 = NumParam(default=10,
                            tex_name=r't_{fu2}',
                            info='Stand time for over frequency 2',
                            non_negative=True,
                            )
 
       # -- protection parameters, voltage
       self.ul1 = NumParam(default=0.88,
                           tex_name='ul1',
                           info='Under voltage shadding point 1',
                           unit='p.u.',
                           )
       self.uu1 = NumParam(default=1.1,
                           tex_name='uu1',
                           info='Over voltage shadding point 1',
                           unit='p.u.',
                           )
       self.ul2 = NumParam(default=0.6,
                           tex_name='ul2',
                           info='Over voltage shadding point 2',
                           unit='p.u.',
                           )
       self.uu2 = NumParam(default=1.2,
                           tex_name='uu2',
                           info='Over voltage shadding point 2',
                           unit='p.u.',
                           )
       self.ul3 = NumParam(default=0.45,
                           tex_name='ul3',
                           info='Under voltage shadding point 3',
                           unit='p.u.',
                           )
       self.ul4 = NumParam(default=0.1,
                           tex_name='ul4',
                           info='Over voltage shadding point 4',
                           unit='p.u.',
                           )
       self.uu4 = NumParam(default=2,
                           tex_name='uu4',
                           info='Over voltage shadding point 4',
                           unit='p.u.',
                           )
 
       self.Tvl1 = NumParam(default=2,
                            tex_name=r't_{vl1}',
                            info='Stand time for under voltage deviation 1',
                            non_negative=True,
                            )
       self.Tvl2 = NumParam(default=1,
                            tex_name=r't_{vl2}',
                            info='Stand time for under voltage deviation 2',
                            non_negative=True,
                            )
       self.Tvl3 = NumParam(default=0.16,
                            tex_name=r't_{vl3}',
                            info='Stand time for under voltage deviation 3',
                            non_negative=True,
                            )
       self.Tvu1 = NumParam(default=1,
                            tex_name=r't_{vu1}',
                            info='Stand time for over voltage deviation 1',
                            non_negative=True,
                            )
       self.Tvu2 = NumParam(default=0.16,
                            tex_name=r't_{vu2}',
                            info='Stand time for over voltage deviation 2',
                            non_negative=True,
                            )
 
class PLK2Model(Model):
   """
   Model implementation of PLK2.
   """
 
   def __init__(self, system, config):
       Model.__init__(self, system, config)
       self.flags.tds = True
       self.group = 'DG'
 
       self.bus = ExtParam(model='DG',
                           src='bus',
                           indexer=self.dev,
                           export=False)
 
       # -- Frequency protection
       self.fn = ExtParam(model='DG',
                          src='fn',
                          indexer=self.dev,
                          export=False)
       # Convert frequency deviation range to p.u.
       self.f = ExtAlgeb(model='FreqMeasurement',
                         src='f',
                         indexer=self.busfreq,
                         export=False,
                         info='Bus frequency',
                         unit='p.u.',
                         )
 
       # Indicatior of frequency deviation
       self.fcl1 = Limiter(u=self.f,
                           lower=self.fl2,
                           upper=self.fl1,
                           tex_name=r'f_{cl1}',
                           info='Frequency comparer for (fl2, fl1)',
                           equal=False,
                           )
       self.fdevl1 = Algeb(v_str='0',
                           e_str='fcl1_zi - fdevl1',
                           info='Frequency  deviation indicator for (fl2, fl1)',
                           tex_name='zs_{Fcl1}',
                           )
 
       self.fcl2 = Limiter(u=self.f,
                           lower=self.fl3,
                           upper=self.fl2,
                           tex_name=r'f_{cl2}',
                           info='Frequency comparer for (fl2, fl1)',
                           equal=False,
                           )
       self.fdevl2 = Algeb(v_str='0',
                           e_str='fcl1_zi - fdevl1',
                           info='Frequency  deviation indicator for (fl2, fl1)',
                           tex_name='zs_{Fcl2}',
                           )
 
       self.fcu1 = Limiter(u=self.f,
                           lower=self.fl3,
                           upper=self.fl2,
                           tex_name=r'f_{cu1}',
                           info='Frequency comparer for (fl2, fl1)',
                           equal=False,
                           )
       self.fdevu1 = Algeb(v_str='0',
                           e_str='fcl1_zi - fdevl1',
                           info='Frequency  deviation indicator for (fl2, fl1)',
                           tex_name='zs_{Fcu1}',
                           )
 
       self.fcu2 = Limiter(u=self.f,
                           lower=self.fl3,
                           upper=self.fl2,
                           tex_name=r'f_{cu2}',
                           info='Frequency comparer for (fl2, fl1)',
                           equal=False,
                           )
       self.fdevu2 = Algeb(v_str='0',
                           e_str='fcl1_zi - fdevl1',
                           info='Frequency  deviation indicator for (fl2, fl1)',
                           tex_name='zs_{Fcu2}',
                           )
 
       # Delayed frequency deviation indicator
       self.freq_devd = Delay(u=self.freq_dev,
                              mode='time',
                              delay=self.Tf.v)
 
       # -- Voltage protection
       self.v = ExtAlgeb(model='Bus',
                         src='v',
                         indexer=self.bus,
                         export=False,
                         info='Bus voltage',
                         unit='p.u.',
                         )
       # Indicatior of voltage deviation
       self.Vcmp = Limiter(u=self.v,
                           lower=self.ul,
                           upper=self.uu,
                           tex_name=r'V_{cmp}',
                           info='Voltage comparator',
                           equal=False,
                           )
       self.Volt_dev = Algeb(v_str='0',
                             e_str='1 - Vcmp_zi - Volt_dev',
                             info='Voltage deviation indicator',
                             tex_name='zs_{Vdev}',
                             )
       # Delayed voltage deviation indicator
       self.Volt_devd = Delay(u=self.Volt_dev,
                              mode='time',
                              delay=self.Tv.v)
 
 
       # -- Lock PVD1 current command
       # freqyency protection
       self.Ipul_f = ExtAlgeb(model='DG',
                              src='Ipul',
                              indexer=self.dev,
                              export=False,
                              e_str='-1000 * Ipul_f * freq_devd_v * freq_dev * fena',
                              info='Current locker from frequency protection',
                              )
       # voltage protection
       self.Ipul_V = ExtAlgeb(model='DG',
                              src='Ipul',
                              indexer=self.dev,
                              export=False,
                              e_str='-1000 * Ipul_V * Volt_devd_v * Volt_dev * Vena',
                              info='Current locker from voltage protection',
                              )
 
 
class PLK2(PLK2Data, PLK2Model):
   """
   DER protection model type 2. PLK stands for Power Lock.
  
   Derived from PLK, followed IEEE-1547.
 
   Frequency (Hz):\n
   (fl3, fl2), Tfl2;\n
   [fl2, fl1), Tfl1;\n
   (fu1, fu2], Tfu1;\n
   (fu2, fu3), Tfu2;\n
   Default:\n
   (50.0, 57.5), 10s;\n
   [57.5, 59.2), 300s;\n
   (60.5, 61.5], 300s;\n
   (61.5, 70.0), 10s.\n
 
   Voltage (p.u.):\n
   (ul4, ul3), Tvl3;\n
   [ul3, ul2), Tvl2;\n
   [ul2, ul1), Tvl1;\n
   [uu1, uu2), Tvu1;\n
   [uu2, uu4), Tvu2.\n
   Default:\n
   (0.10, 0.45), 0.16s;\n
   [0.45, 0.60), 1s;\n
   [0.60, 0.88), 2s;\n
   [1.10, 1.20), 1s;\n
   [1.20, 2.00), 0.16s.\n
 
   Target device (limited to DG group) ``Ipul`` will drop to zero immediately
   when frequency/voltage protection is triggered. Once the lock is released,
   ``Ipul`` will return to normal immediately.
 
   ``fena`` and ``Vena`` are protection enabling parameters. 1 is on and 0 is off.
   """
 
   def __init__(self, system, config):
       PLK2Data.__init__(self)
       PLK2Model.__init__(self, system, config)
 

