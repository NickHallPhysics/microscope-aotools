#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2018 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>, Josh Edwards
## <Josh.Edwards222@gmail.com> & Jacopo Antonello
## <jacopo.antonello@dpag.ox.ac.uk>
##
## microAO is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## microAO is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with microAO.  If not, see <http://www.gnu.org/licenses/>.

#Import required packs
import numpy as np
import Pyro4
import time


from microscope.devices import Device
from microscope.devices import TriggerType
from microscope.devices import TriggerMode

class AdaptiveOpticsDevice(Device):
    """Class for the adaptive optics device

    This class requires a mirror and a camera. Everything else is generated
    on or after __init__"""
    
    _CockpitTriggerType_to_TriggerType = {
    "SOFTWARE" : TriggerType.SOFTWARE,
    "RISING_EDGE" : TriggerType.RISING_EDGE,
    "FALLING_EDGE" : TriggerType.FALLING_EDGE,
    }

    _CockpitTriggerModes_to_TriggerModes = {
    "ONCE" : TriggerMode.ONCE,
    "START" : TriggerMode.START,
    }

    def __init__(self, wavefront_uri, mirror_uri, **kwargs):
        # Init will fail if devices it depends on aren't already running, but
        # deviceserver should retry automatically.
        super(AdaptiveOpticsDevice, self).__init__(**kwargs)
        # Wavefront sensor. Must support soft_trigger for now.
        self.wavefront_camera = Pyro4.Proxy('PYRO:%s@%s:%d' %(wavefront_uri[0].__name__,
                                                    wavefront_uri[1], wavefront_uri[2]))
        self.wavefront_camera.enable()
        # Deformable mirror device.
        self.mirror = Pyro4.Proxy('PYRO:%s@%s:%d' %(mirror_uri[0].__name__,
                                                mirror_uri[1], mirror_uri[2]))
        #self.mirror.set_trigger(TriggerType.RISING_EDGE) #Set trigger type to rising edge
        self.numActuators = self.mirror.n_actuators
        # Region of interest (i.e. pupil offset and radius) on camera.
        self.roi = None
        #Mask for the interferometric data
        self.mask = None
        #Mask to select phase information
        self.fft_filter = None
        #Control Matrix
        self.controlMatrix = None
        #System correction
        self.flat_actuators_sys = np.zeros(self.numActuators)
        #Last applied actuators values
        self.last_actuator_values = None
        # Last applied actuators pattern
        self.last_actuator_patterns = None

        ##We don't use all the actuators. Create a mask for the actuators outside
        ##the pupil so we can selectively calibrate them. 0 denotes actuators at
        ##the edge, i.e. outside the pupil, and 1 denotes actuators in the pupil

        #Use this if all actuators are being used
        #self.pupil_ac = np.ones(self.numActuators)

        #Preliminary mask for DeepSIM
        self.pupil_ac = np.ones(self.numActuators)

        try:
            assert np.shape(self.pupil_ac)[0] == self.numActuators
        except:
            raise Exception("Length mismatch between pupil mask (%i) and "
                            "number of actuators (%i). Please provide a mask "
                            "of the correct length" %(np.shape(self.pupil_ac)[0],
                                                      self.numActuators))

    def _on_shutdown(self):
        pass

    def initialize(self, *args, **kwargs):
        pass

    @Pyro4.expose
    def set_trigger(self, cp_ttype, cp_tmode):
        ttype = self._CockpitTriggerType_to_TriggerType[cp_ttype]
        tmode = self._CockpitTriggerModes_to_TriggerModes[cp_tmode]
        self.mirror.set_trigger(ttype, tmode)

    @Pyro4.expose
    def get_pattern_index(self):
        return self.mirror.get_pattern_index()

    @Pyro4.expose
    def get_n_actuators(self):
        return self.numActuators

    @Pyro4.expose
    def send(self, values):
        #Need to normalise patterns because general DM class expects 0-1 values
        values[values > 1.0] = 1.0
        values[values < 0.0] = 0.0

        try:
            self.mirror.apply_pattern(values)
        except Exception as e:
            self._logger.info(e)

        self.last_actuator_values = values

    @Pyro4.expose
    def get_last_actuator_values(self):
        return self.last_actuator_values

    @Pyro4.expose
    def queue_patterns(self, patterns):
        self._logger.info("Queuing patterns on DM")

        # Need to normalise patterns because general DM class expects 0-1 values
        patterns[patterns > 1.0] = 1.0
        patterns[patterns < 0.0] = 0.0

        try:
            self.mirror.queue_patterns(patterns)
        except Exception as e:
            self._logger.info(e)

        self.last_actuator_patterns = patterns

    @Pyro4.expose
    def get_last_actuator_patterns(self):
        return self.last_actuator_patterns

    @Pyro4.expose
    def set_roi(self, y0, x0, radius):
        self.roi = (y0, x0, radius)
        try:
            assert self.roi is not None
        except:
            raise Exception("ROI assignment failed")

        #Mask will need to be reconstructed as radius has changed
        self.mask = self.make_mask(radius)
        try:
            assert self.mask is not None
        except:
            raise Exception("Mask construction failed")

        #Fourier filter should be erased, as it's probably wrong. 
        ##Might be unnecessary
        self.fft_filter = None
        return

    @Pyro4.expose
    def get_roi(self):
        if np.any(self.roi) is None:
            raise Exception("No region of interest selected. Please select a region of interest")
        else:
            return self.roi

    @Pyro4.expose
    def get_fourierfilter(self):
        if np.any(self.fft_filter) is None:
            raise Exception("No Fourier filter created. Please create one.")
        else:
            return self.fft_filter

    @Pyro4.expose
    def get_controlMatrix(self):
        if np.any(self.controlMatrix) is None:
            raise Exception("No control matrix calculated. Please calibrate the mirror")
        else:
            return self.controlMatrix


    @Pyro4.expose
    def set_controlMatrix(self,controlMatrix):
        self.controlMatrix = controlMatrix
        return

    @Pyro4.expose
    def reset(self):
        self._logger.info("Resetting DM")
        self.send(np.zeros(self.numActuators) + 0.5)


    @Pyro4.expose
    def acquire_raw(self):
        self.acquiring = True
        while self.acquiring == True:
            try:
                data_raw, timestamp = self.wavefront_camera.grab_next_data()
                self.acquiring = False
            except Exception as e:
                if str(e) == str("ERROR 10: Timeout"):
                    self._logger.info("Recieved Timeout error from camera. Waiting to try again...")
                    time.sleep(1)
                else:
                    self._logger.info(type(e))
                    self._logger.info("Error is: %s" %(e))
                    raise e
        return data_raw

    @Pyro4.expose
    def acquire(self):
        self.acquiring = True
        while self.acquiring == True:
            try:
                data_raw, timestamp = self.wavefront_camera.grab_next_data()
                self.acquiring = False
            except Exception as e:
                if str(e) == str("ERROR 10: Timeout"):
                    self._logger.info("Recieved Timeout error from camera. Waiting to try again...")
                    time.sleep(1)
                else:
                    self._logger.info(type(e))
                    self._logger.info("Error is: %s" %(e))
                    raise e
        if np.any(self.roi) is None:
            data = data_raw
        else:
            data_cropped = np.zeros((self.roi[2] * 2, self.roi[2] * 2), dtype=float)
            data_cropped[:, :] = data_raw[self.roi[0] - self.roi[2]:self.roi[0] + self.roi[2],
                                 self.roi[1] - self.roi[2]:self.roi[1] + self.roi[2]]
            if np.any(self.mask) is None:
                self.mask = self.make_mask(self.roi[2])
                data = data_cropped
            else:
                data = data_cropped * self.mask
        return data

    def make_mask(self, radius):
        diameter = radius * 2
        self.mask = np.sqrt((np.arange(-radius, radius) ** 2).reshape((diameter, 1)) + (np.arange(-radius, radius) ** 2)) < radius
        return self.mask

    @Pyro4.expose
    def phaseunwrap(self, image = None):
        #Here is where interferometric data is unwrapped
        return unwrapped_image


    @Pyro4.expose
    def getzernikemodes(self, unwrapped_image):
        #Here you should fit Zernike polynomials to your unwrapped image and return the coefficients
        return coef

    @Pyro4.expose
    def createcontrolmatrix(self, imageStack, noZernikeModes, pokeSteps):
        #Here you should create a control matrix for your DM
        return self.controlMatrix


    def measurequalitymetric(self, bead_image):
        #Here you should measure the quality metric of bead images (or any other sample, in theory)
        return quality_matric

    def findoptimalzernikemode_sensorless(self, images, zernike_mode_amplitude_applied):
        #Here you should take in an image stack where one zernike mode's amplitude has been varied,
        #measure the quality metric for each image and then fit those to a parabola to obtain the optimal amplitude
        #for this Zernike mode
        return optimal_amplitude

    def findalloptimalzernikemode_sensorless(selfs, images_all, all_zernike_mode_amplitude_applied):
        #Here you should recieve all the images and all the zernike modes applied, break them down into
        #individual zernike modes
        return