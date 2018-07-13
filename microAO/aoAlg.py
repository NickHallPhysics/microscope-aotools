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
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import tukey, gaussian
import aotools
import scipy.stats as stats
from skimage.restoration import unwrap_phase
from scipy.integrate import trapz
import time

import logging

class AdaptiveOpticsFunctions:
    def __init__(self):
        logging.basicConfig(filename='example.log',level=logging.DEBUG)

        self.mask = None
        self.fft_filter = None
        self.controlMatrix = None

    def make_mask(self, radius):
        diameter = radius * 2
        self.mask = np.sqrt((np.arange(-radius,radius)**2).reshape((diameter,1)) + (np.arange(-radius,radius)**2)) < radius
        return self.mask


    def bin_ndarray(self, ndarray, new_shape, operation='sum'):
        """

        Function acquired from Stack Overflow: https://stackoverflow.com/a/29042041. Stack Overflow or other Stack Exchange
        sites is cc-wiki (aka cc-by-sa) licensed and requires attribution.

        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.

        Number of output dimensions must match number of input dimensions and
            new axes must divide old ones.

        Example
        -------

        m = np.arange(0,100,1).reshape((10,10))
        n = bin_ndarray(m, new_shape=(5,5), operation='sum')
        print(n)

        [[ 22  30  38  46  54]
         [102 110 118 126 134]
         [182 190 198 206 214]
         [262 270 278 286 294]
         [342 350 358 366 374]]

        """
        operation = operation.lower()
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                               new_shape))
        compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                      ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))
        return ndarray

    def mgcentroid(self, myim, mythr=0.0):
        assert(myim.dtype == np.float)

        myn1, myn2 = myim.shape
        myxx1, myxx2 = np.meshgrid(range(1, myn1 + 1), range(1, myn2 + 1))
        myim[myim < mythr] = 0
        mysum1 = np.sum((myxx1*myim).ravel())
        mysum2 = np.sum((myxx2*myim).ravel())
        mymass = np.sum(myim.ravel())
        return int(np.round(mysum1/mymass)), int(np.round(mysum2/mymass))

    def fft_filter(self,image, region=None):
        #Convert image to array and float
        data = np.asarray(image)

        if region is None:
            region = int(data.shape[0]/8.0)

        #Apply tukey window
        fringes = np.fft.fftshift(data)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1)*tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        #Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        #Remove center section to allow finding of 1st order point
        fftarray = np.fft.fftshift(fftarray)
        find_cent = [int(fftarray.shape[1]/2),int(fftarray.shape[0]/ 2)]
        fftarray[find_cent[1]-region:find_cent[1]+region,find_cent[0]-region:find_cent[0]+region]=0.00001+0j

        #Find approximate position of first order point
        test_point = np.argmax(fftarray)
        test_point= [int(test_point%fftarray.shape[1]),int(test_point/fftarray.shape[1])]

        #Find first order point
        maxpoint = np.zeros(np.shape(test_point),dtype = int)
        maxpoint[:] = test_point[:]
        window = np.zeros((50,50))

        weight_1D = gaussian(50,50)
        weight = np.outer(weight_1D,weight_1D.T)
        weight = weight*(weight>weight[24,49])

        for ii in range(10):
            window[:,:] = np.log(abs(fftarray[maxpoint[1]-25:maxpoint[1]+25,maxpoint[0]-25:maxpoint[0]+25]))
            thresh = np.max(window) - 5
            CoM = np.zeros((1,2))
            window[window < thresh] = 0
            window[:,:] = window[:,:] * weight[:,:]
            CoM[0,:] = np.round(center_of_mass(window))
            maxpoint[0] = maxpoint[0] - 25 + int(CoM[0,1])
            maxpoint[1] = maxpoint[1] - 25 + int(CoM[0,0])

        self.fft_filter = np.zeros(np.shape(fftarray))
        mask_di = min(int(data.shape[0]*(5.0/16.0)), (maxpoint[0]-maxpoint[0]%2), (maxpoint[1]-maxpoint[1]%2))

        x = np.sin(np.linspace(0, np.pi, mask_di))**2
        fourier_mask = np.outer(x,x.T)

        y_min = maxpoint[1]-int(np.floor((mask_di/2.0)))
        y_max = maxpoint[1]+int(np.ceil((mask_di/2.0)))
        x_min = maxpoint[0]-int(np.floor((mask_di/2.0)))
        x_max = maxpoint[0]+int(np.ceil((mask_di/2.0)))

        self.fft_filter[y_min:y_max,x_min:x_max] = fourier_mask
        return self.fft_filter

    def phase_unwrap(self,image):
        #Convert image to array and float
        data = np.asarray(image)

        #Apply tukey window
        fringes = np.fft.fftshift(data)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1)*tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        #Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        #Apply Fourier filter
        M = np.fft.fftshift(self.fft_filter)
        fftarray_filt = fftarray * M
        fftarray_filt = np.fft.fftshift(fftarray_filt)

        #Roll data to the centre
        g0, g1 = self.mgcentroid(self.fft_filter) - np.round(fftarray_filt.shape[0]//2)
        fftarray_filt = np.roll(fftarray_filt, -g0, axis=1)
        fftarray_filt = np.roll(fftarray_filt, -g1, axis=0)

        #Convert to real space
        fftarray_filt_shift = np.fft.fftshift(fftarray_filt)
        complex_phase = np.fft.fftshift(np.fft.ifft2(fftarray_filt_shift))

        #Find phase data by taking 2d arctan of imaginary and real parts
        phaseorder1 = np.zeros(complex_phase.shape)
        phaseorder1[:,:] = np.arctan2(complex_phase.imag,complex_phase.real)

        #Mask out edge region to allow unwrap to only use correct region
        phaseorder1mask = phaseorder1 * self.mask

        #Perform unwrap
        phaseorder1unwrap = unwrap_phase(phaseorder1mask)
        out = phaseorder1unwrap * self.mask
        return out

    def get_zernike_modes(self, image_unwrap, noZernikeModes, resize_dim = 128):
        #Resize image
        original_dim = int(np.shape(image_unwrap)[0])
        while original_dim%resize_dim is not 0:
            resize_dim -= 1

        if resize_dim < original_dim/resize_dim:
            resize_dim = original_dim/resize_dim

        image_resize = self.bin_ndarray(image_unwrap, new_shape=(resize_dim,resize_dim), operation='mean')

        #Calculate Zernike mode
        zcoeffs_dbl = []
        num_pixels = np.count_nonzero(aotools.zernike(1, resize_dim))
        for i in range(1,(noZernikeModes+1)):
            intermediate = trapz(image_resize * aotools.zernike(i, resize_dim))
            zcoeffs_dbl.append(trapz(intermediate) / (num_pixels))
        coef = np.asarray(zcoeffs_dbl)
        return coef

    def create_control_matrix(self, imageStack, numActuators, noZernikeModes, pokeSteps, pupil_ac = None, threshold = 0.005):
        slopes = np.zeros(noZernikeModes)
        intercepts = np.zeros(noZernikeModes)
        r_values = np.zeros(noZernikeModes)
        p_values = np.zeros(noZernikeModes)
        std_errs = np.zeros(noZernikeModes)

        # Define variables
        try:
            assert type(imageStack) is np.ndarray
        except:
            logging.ERROR("Expected numpy.ndarray input data type, got %s" %type(imageStack))
        [noImages, x, y] = np.shape(imageStack)
        numPokeSteps = len(pokeSteps)

        C_mat = np.zeros((noZernikeModes,numActuators))
        all_zernikeModeAmp = np.ones((noImages,noZernikeModes))
        offsets = np.zeros((noZernikeModes,numActuators))
        P_tests = np.zeros((noZernikeModes,numActuators))

        assert x == y
        edge_mask = np.sqrt((np.arange(-x,x)**2).reshape((x*2,1)) + (np.arange(-x,x)**2)) < x-3

        # Here the each image in the image stack (read in as np.array), centre and diameter should be passed to the unwrap
        # function to obtain the Zernike modes for each one. For the moment a set of random Zernike modes are generated.
        for ii in range(numActuators):
            if pupil_ac[ii] == 1:
                pokeSteps_trimmed_list = []
                zernikeModeAmp_list = []
                #Get the amplitudes of each Zernike mode for the poke range of one actuator
                for jj in range(numPokeSteps):
                    curr_calc = (ii * numPokeSteps) + jj + 1
                    logging.INFO("Calculating Zernike modes %d/%d..." %(curr_calc, noImages))
                    image_unwrap = self.phase_unwrap(imageStack[((ii * numPokeSteps) + jj),:,:])
                    diff_image = abs(np.diff(np.diff(image_unwrap,axis=1),axis=0)) * edge_mask[:-1,:-1]
                    if np.any(diff_image > 2*np.pi):
                        logging.INFO("Unwrap image %d/%d contained discontinuites" %(curr_calc, noImages))
                        logging.INFO("Zernike modes %d/%d not calculates" %(curr_calc, noImages))
                    else:
                        pokeSteps_trimmed_list.append(pokeSteps[jj])
                        logging.INFO("Calculating Zernike modes %d/%d..." %(curr_calc, noImages))
                        curr_amps = self.get_zernike_modes(image_unwrap, noZernikeModes)
                        thresh_amps = curr_amps * (abs(curr_amps)>0.5)
                        zernikeModeAmp_list.append(thresh_amps)
                        all_zernikeModeAmp[(curr_calc-1),:] = thresh_amps
                        logging.INFO("Zernike modes %d/%d calculated" %(curr_calc, noImages))

                pokeSteps_trimmed = np.asarray(pokeSteps_trimmed_list)
                zernikeModeAmp = np.asarray(zernikeModeAmp_list)

                #Fit a linear regression to get the relationship between actuator position and Zernike mode amplitude
                for kk in range(noZernikeModes):
                    logging.INFO("Fitting regression %d/%d..." % (kk+1, noZernikeModes))
                    try:
                        slopes[kk],intercepts[kk],r_values[kk],p_values[kk],std_errs[kk] = \
                            stats.linregress(pokeSteps_trimmed,zernikeModeAmp[:,kk])
                    except Exception as e:
                        logging.ERROR(e)
                    logging.INFO("Regression %d/%d fitted" % (kk + 1, noZernikeModes))

                #Input obtained slopes as the entries in the control matrix
                C_mat[:,ii] = slopes[:]
                offsets[:,ii] = intercepts[:]
                P_tests[:,ii] = p_values[:]
            else:
                logging.INFO("Actuator %d is not in the pupil and therefore skipped" % (ii))
        logging.INFO("Computing Control Matrix")
        self.controlMatrix = np.linalg.pinv(C_mat, rcond=threshold)
        logging.INFO("Control Matrix computed")
        return self.controlMatrix

    def ac_pos_from_zernike(self, applied_z_modes, numActuators, offset = None):
        if int(np.shape(applied_z_modes)[0]) < int(np.shape(self.controlMatrix)[1]):
            pad_length = int(np.shape(applied_z_modes)[0]) - int(np.shape(self.controlMatrix)[1])
            np.pad(applied_z_modes, (0,pad_length), 'constant')
        elif int(np.shape(applied_z_modes)[0]) > int(np.shape(self.controlMatrix)[1]):
            applied_z_modes = applied_z_modes[:int(np.shape(self.controlMatrix)[1])]
        else:
            pass

        actuator_pos = np.zeros(numActuators)
        if offset is not None:
            actuator_pos[:] = np.dot(self.controlMatrix, applied_z_modes) + offset
        else:
            actuator_pos[:] = np.dot(self.controlMatrix, applied_z_modes)
        return actuator_pos