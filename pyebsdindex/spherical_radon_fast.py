"""This software was developed by employees of the US Naval Research Laboratory (NRL), an
agency of the Federal Government. Pursuant to title 17 section 105 of the United States
Code, works of NRL employees are not subject to copyright protection, and this software
is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
responsibility whatsoever for its use by other parties, and makes no guarantees,
expressed or implied, about its quality, reliability, or any other characteristic. We
would appreciate acknowledgment if the software is used. To the extent that NRL may hold
copyright in countries other than the United States, you are hereby granted the
non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
works and distribute this software, in any medium, or authorize others to do so on your
behalf, on a royalty-free basis throughout the world. You may improve, modify, and
create derivative works of the software or any portion of the software, and you may copy
and distribute such modifications or works. Modified works should carry a notice stating
that you changed the software and should note the date and nature of any such change.
Please explicitly acknowledge the US Naval Research Laboratory as the original source.
This software can be redistributed and/or modified freely provided that any derivative
works bear some notice that they are derived from it, and any modified versions bear
some notice that they have been modified.

Author: David Rowenhorst;
The US Naval Research Laboratory Date: 21 Aug 2020"""

from os import environ
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from numba import jit, prange
import numpy as np
import time
RADEG = 180.0/np.pi
DEGRAD = np.pi/180.0



class SphericalRadon:
  def __init__(self, image=None, imageDim=None,
               nTheta=90, nPhi=180,
               thetaRange =  [45,135],
               phiRange = [-90, 90], # [-45,45],
               vendor='EDAX'):
    self.nTheta = nTheta
    self.nPhi = nPhi
    self.thetaRange = np.array(thetaRange)
    self.phiRange = np.array(phiRange)

    self.indexPlan = None

    self.vendor = vendor

    if (image is None) and (imageDim is None):
      self.theta = None
      self.phi = None
      self.imDim = None

    else:
      if image is not None:
        self.imDim = np.asarray(image.shape[-2:])
      else:
        self.imDim = np.asarray(imageDim[-2:])
      self.radon_plan_setup(imageDim=self.imDim, nTheta=self.nTheta, nPhi=self.nPhi,
                            thetaRange = self.thetaRange,phiRange= self.phiRange)


  def set_theta_range(self, PC=[0.5, 0.5, 0.5]):
    # function that will look at the pattern center(s) and make a decision on the range of
    # theta values to use.  It will also set (nTheta) so that it matches the angular resolution of
    # phi.  
    pc_px = self.convert_pc_detector(PC)
    pcx = np.max(pc_px[:, 0])
    pcy = np.max(pc_px[:, 1])
    pcz = np.min(pc_px[:, 2])
    nX = np.float32(self.imDim[1])
    nY = np.float32(self.imDim[0])

    thetas = np.array([pcx**2 + pcy**2,
                       (nX-pcx)**2 + pcy**2,
                       (nX-pcx)**2 + (nY-pcy)**2,
                       pcx**2 + (nY-pcy)**2])

    thetas = np.sqrt(thetas)
    thetas /= pcz
    thetas = 90 - np.degrees(np.arctan(thetas))
    trange = np.array([np.floor(thetas.min()), np.ceil(180 - thetas.min())])

    self.thetaRange = trange
    dphi = (np.max(self.phiRange) - np.min(self.phiRange)) / self.nPhi
    self.nTheta = np.round((np.max(self.thetaRange) - np.min(self.thetaRange)) * dphi).astype(np.int64)
    #print(dphi, self.nTheta)
    self.radon_plan_setup(imageDim=self.imDim)

  def radon_plan_setup(self, image=None, imageDim=None, nTheta=None, nPhi=None, thetaRange = None,phiRange= None):
    if (image is None) and (imageDim is not None):
      imDim = np.asarray(imageDim, dtype=np.int64)
    elif (image is not None):
      imDim =  np.shape(image)[-2:] # this will catch if someone sends in a [1 x N x M] image
    else:
      return -1
    imDim = np.asarray(imDim)
    self.imDim = imDim
    if (nTheta is not None) : self.nTheta = nTheta
    if (nPhi is not None): self.nPhi = nPhi
    if (thetaRange is not None) : self.thetaRange = np.array(thetaRange)
    if (phiRange is not None): self.phiRange = np.array(phiRange)

    self.theta = np.arange(self.nTheta, dtype = np.float32)*(self.thetaRange.max()-self.thetaRange.min())/self.nTheta
    self.theta += self.thetaRange.min()
    self.phi = np.arange(self.nPhi, dtype=np.float32) * (
          self.phiRange.max() - self.phiRange.min()) / self.nPhi
    self.phi += self.phiRange.min()

    #define an array of spherical points, here denoted as hkl (this is in the refernce frame, not crystal frame).

    self.hkl = np.zeros((self.nTheta, self.nPhi, 3), dtype=np.float32)
    self.hkl[:, :, 0] = np.sin(np.radians(self.theta)).reshape(self.nTheta, 1)
    self.hkl[:, :, 1] = np.sin(np.radians(self.theta)).reshape(self.nTheta, 1)
    self.hkl[:, :, 2] = np.cos(np.radians(self.theta)).reshape(self.nTheta, 1)

    self.hkl[:, :, 0] *= np.cos(np.radians(self.phi))
    self.hkl[:, :, 1] *= np.sin(np.radians(self.phi))


  def convert_pc_detector(self, pc):
    # Helper function to convert vendors PC values to pixels on the detector.  This assumes that the detector image
    # origin is in the upper left corner.
    ven = str.upper(self.vendor)
    nX = np.float32( self.imDim[1] )
    nY = np.float32( self.imDim[0] )

    pctemp = np.atleast_2d(pc)
    pc_px = np.zeros((pctemp.shape[0], 3), dtype=np.float32)

    if ven in ['EDAX', 'OXFORD']:
      pc_px[:,0] = nX * pctemp[:,0]
      pc_px[:, 1] = nY - nX * pctemp[:, 1]
      pc_px[:, 2] = nX*pctemp[:, 2]
    if ven in ['KIKUCHIPY', 'BRUKER']:
      pc_px[:, 0] = nX * pctemp[:, 0]
      pc_px[:, 1] = nY  * pctemp[:, 1]
      pc_px[:, 2] = nY * pctemp[:, 2]

    if ven in ['EMSoft']:
      pc_px[:, 0] = nX*0.5 + pctemp[:, 0]
      pc_px[:, 1] = nY - (nY * 0.5 + pctemp[:, 1])
      pc_px[:, 2] = pctemp[:,2]/pctemp[:,3]
    return pc_px

  def radon_fast(self, imageIn, PC = [0.5, 0.5, 0.5],  padding = np.array([0,0]), fixArtifacts = False, background = None):
    #plt.isinteractive()
    tic = timer()
    shapeIm = np.shape(imageIn)
    pctemp = np.atleast_2d(np.asarray(PC, dtype= np.float32))

    if imageIn.ndim == 2:
      reform = True
      image = imageIn[np.newaxis, : ,:]
    else:
      reform = False
      image = imageIn

    nIm = image.shape[0]
    nPC = np.atleast_2d(PC).shape[0]
    pc_px = self.convert_pc_detector(pctemp)

    if nPC < nIm:
      pc_px = np.tile(pc_px[0,:], nIm).reshape(nIm,3)

    if background is not None:
      image = imageIn - np.atleast_3d(background)


    nX = shapeIm[-1]
    nY = shapeIm[-2]

    #radon = np.zeros([nIm, self.nRho, self.nTheta], dtype=np.float32)
    radon = np.zeros([self.nTheta + 2 * padding[0],self.nPhi + 2 * padding[1], nIm],dtype=np.float32)
    shpRdn = radon.shape

    xythresh = np.pi/4.0
    x = np.arange(nX, dtype = np.int64)
    y = np.arange(nY, dtype = np.int64)

    for i in np.arange(self.nTheta):
      for j in np.arange(self.nPhi):

        h = self.hkl[i,j,0]
        k = self.hkl[i,j,1]
        l = self.hkl[i,j,2]
        if np.arctan2(abs(h), abs(k))  < xythresh:
          dydx = h/k
          for ii in np.arange(nIm):
            ystart =  pc_px[ii, 1] - (h * pc_px[ii, 0] - l * pc_px[ii, 2]) / k
            yy = np.round(ystart + dydx*x.astype(np.float32)).astype(np.int64)
            wh = np.flatnonzero(np.asarray((yy > 0) & (yy < nY)))
            if wh.shape[0] > 0:
              radon[i+padding[0], j + padding[1], ii] = np.mean(image[ii, yy[wh], x[wh]]).astype(np.float32)
        else:
          dxdy = k/h
          for ii in np.arange(nIm):
            xstart = (-1.0 * l * pc_px[ii,2] -k * pc_px[ii,1])/h + pc_px[ii,0]
            xx = np.round(xstart+dxdy*y.astype(np.float32)).astype(np.int64)
            wh = np.flatnonzero(np.asarray((xx > 0) & (xx < nX)))
            if wh.shape[0] > 0:
              radon[i+padding[0], j+padding[1], ii] = np.mean(image[ii, y[wh], xx[wh]]).astype(np.float32)#/np.float32(wh.shape[0])
    if reform==True:
      image = image.reshape(shapeIm)


    return radon



  @staticmethod
  @jit(nopython=True, fastmath=True, cache=True, parallel=False)
  def rdn_loops(images,index,nIm,nPx,indxdim,radon, padding):
    nRho = indxdim[0]
    nTheta = indxdim[1]
    nIndex = indxdim[2]
    #counter = np.zeros((nRho, nTheta, nIm), dtype=np.float32)
    count = 0.0
    sum = 0.0
    for q in prange(nIm):
      #radon[:,:,q] = np.mean(images[q*nPx:(q+1)*nPx])
      imstart = q*nPx
      for i in range(nRho):
        ip = i+padding[0]
        for j in range(nTheta):
          jp = j+padding[1]
          count = 0.0
          sum = 0.0
          for k in range(nIndex):
            indx1 = index[i,j,k]
            if (indx1 >= nPx):
              break
            #radon[q, i, j] += images[imstart+indx1]
            sum += images[imstart + indx1]
            count += 1.0
          #if count >= 1.0:
            #counter[ip,jp, q] = count
          radon[ip,jp,q] = sum/(count + 1.0e-12)
    #return counter

  def radon2pole(self,bandData,PC=None,vendor='EDAX'):
    # Following Krieger-Lassen1994 eq 3.1.6 //figure 3.1.1
    if PC is None:
      PC = np.array([0.471659,0.675044,0.630139])
    ven = str.upper(vendor)

    nPats = bandData.shape[0]
    nBands = bandData.shape[1]

    # This translation from the Radon to theta and rho assumes that the first pixel read
    # in off the detector is in the bottom left corner. -- No longer the assumption --- see below.
    # theta = self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1], dtype=np.int)]/RADEG
    # rho = self.radonPlan.rho[np.array(bandData['aveloc'][:, :, 0], dtype=np.int)]

    # This translation from the Radon to theta and rho assumes that the first pixel read
    # in off the detector is in the top left corner.

    #theta = np.pi - self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1],dtype=np.int64)] / RADEG
    #rho = -1.0 * self.radonPlan.rho[np.array(bandData['aveloc'][:,:,0],dtype=np.int64)]

    theta =  np.pi - np.interp(bandData['aveloc'][:,:,1], np.arange(self.nTheta), self.theta) / RADEG
    rho = -1.0 * np.interp(bandData['aveloc'][:,:,0], np.arange(self.nRho), self.rho)
    bandData['theta'][:] = theta
    bandData['rho'][:] = rho

    # from this point on, we will assume the image origin and t-vector (aka pattern center) is described
    # at the bottom left of the pattern
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    pctemp =  np.asfarray(PC).copy()
    shapet = pctemp.shape
    if ven != 'EMSOFT':
      if len(shapet) < 2:
        pctemp = np.tile(pctemp, nPats).reshape(nPats,3)
      else:
        if shapet[0] != nPats:
          pctemp = np.tile(pctemp[0,:], nPats).reshape(nPats,3)
      t = pctemp
    else: # EMSOFT pc to ebsdindex needs four numbers for PC
      if len(shapet) < 2:
        pctemp = np.tile(pctemp, nPats).reshape(nPats,4)
      else:
        if shapet[0] != nPats:
          pctemp = np.tile(pctemp[0,:], nPats).reshape(nPats,4)
      t = pctemp[:,0:3]
      t[:,2] /= pctemp[:,3] # normalize by pixel size



    dimf = np.array(self.imDim, dtype=np.float32)
    if ven in ['EDAX', 'OXFORD']:
      t *= np.array([dimf[1], dimf[1], -dimf[1]])
    if ven == 'EMSOFT':
      t[:, 0] *= -1.0
      t += np.array([dimf[1] / 2.0, dimf[0] / 2.0, 0.0])
      t[:, 2] *= -1.0
    if ven in ['KIKUCHIPY', 'BRUKER']:
      t *=  np.array([dimf[1], dimf[0], -dimf[0]])
      t[:, 1] = dimf[0] - t[:, 1]
    # describes the translation from the bottom left corner of the pattern image to the point on the detector
    # perpendicular to where the beam contacts the sample.


    t = np.tile(t.reshape(nPats,1, 3), (1, nBands,1))

    r = np.zeros((nPats, nBands, 3), dtype=np.float32)
    r[:,:,0] = -1*stheta
    r[:,:,1] = ctheta # now defined as r_v

    p = np.zeros((nPats, nBands, 3), dtype=np.float32)
    p[:,:,0] = rho*ctheta # get a point within the band -- here it is the point perpendicular to the image center.
    p[:,:,1] = rho*stheta
    p[:,:,0] += dimf[1] * 0.5 # now convert this with reference to the image origin.
    p[:,:,1] += dimf[0] * 0.5 # this is now [O_vP]_v in Eq 3.1.6

    #n2 = p - t.reshape(1,1,3)
    n2 = p - t
    n = np.cross(r.reshape(nPats*nBands, 3), n2.reshape(nPats*nBands, 3) )
    norm = np.linalg.norm(n, axis=1)
    n /= norm.reshape(nPats*nBands, 1)
    n = n.reshape(nPats, nBands, 3)
    return n