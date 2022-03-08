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
from pathlib import PurePath
import platform
import tempfile
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numba
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_dilation as scipy_grey_dilation

from pyebsdindex import openclparam, radon_fast




tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
tempdir = tempdir.joinpath('numba')
environ["NUMBA_CACHE_DIR"] = str(tempdir)

RADEG = 180.0/np.pi



class BandDetect:
  def __init__(
    self,
    patterns=None,
    patDim=None,
    nTheta=180,
    nRho=90,
    tSigma=None,
    rSigma=None,
    rhoMaskFrac=0.1,
    nBands=9
):
    self.patDim = None
    self.nTheta = nTheta
    self.nRho = nRho
    self.dTheta = None
    self.dRho = None
    self.rhoMax = None
    self.radonPlan = None
    self.rdnNorm = None
    self.tSigma = tSigma
    self.rSigma = rSigma
    self.kernel = None
    self.peakPad = np.array([11, 11])
    self.padding = np.array([11, 11])
    self.rhoMaskFrac = rhoMaskFrac

    self.nBands = nBands
    self.EDAXIQ = False
    self.backgroundsub = None

    self.dataType = np.dtype([('id', np.int32), ('max', np.float32), \
                    ('maxloc', np.float32, (2)), ('avemax', np.float32), ('aveloc', np.float32, (2)),\
                    ('pqmax', np.float32), ('width', np.float32), ('valid', np.int8)])


    if (patterns is None) and (patDim is None):
      pass
    else:
      if (patterns is not None):
        self.patDim = np.asarray(patterns.shape[-2:])
      else:
        self.patDim = np.asarray(patDim)
      self.band_detect_setup(patterns, self.patDim,self.nTheta,self.nRho,\
                self.tSigma, self.rSigma,self.rhoMaskFrac,self.nBands)

  def band_detect_setup(self, patterns=None,patDim=None,nTheta=None,nRho=None,\
                      tSigma=None, rSigma=None,rhoMaskFrac=None,nBands=None):
    p_dim = None
    recalc_radon = False
    recalc_masks = False
    if (patterns is None) and (patDim is not None):
      p_dim = np.asarray(patDim, dtype=np.int64)
    if patterns is not None:
      p_dim = np.shape(patterns)[-2:]  # this will catch if someone sends in a [1 x N x M] image
    if p_dim is not None:
      if self.patDim is None:
        recalc_radon = True
        self.patDim = p_dim

      elif np.sum(np.abs(self.patDim[-2:]-p_dim[-2:]), dtype=np.int64) != 0:
        recalc_radon = True
        self.patDim = p_dim

    if nTheta is not None:
      self.nTheta = nTheta
      recalc_radon = True
      recalc_masks = True

    if self.nTheta is not None:
      self.dTheta = 180.0/self.nTheta


    if nRho is not None:
      self.nRho = nRho
      self.dRho = 180. / self.nRho
      recalc_radon = True
      recalc_masks = True

    if self.dRho is None:
      recalc_radon = True

    if recalc_radon == True:
      self.rhoMax = 0.5 * np.float32(self.patDim.min())
      self.dRho = self.rhoMax/np.float32(self.nRho)
      self.radonPlan = radon_fast.Radon(imageDim=self.patDim, nTheta=self.nTheta, nRho=self.nRho, rhoMax=self.rhoMax)
      temp = np.ones(self.patDim[-2:], dtype=np.float32)
      back = self.radonPlan.radon_faster(temp,fixArtifacts=True)
      back = (back > 0).astype(np.float32) / (back + 1.0e-12)
      self.rdnNorm = back


    if tSigma is not None:
      self.tSigma = tSigma
      recalc_masks = True
    if rSigma is not None:
      self.rSigma = rSigma
      recalc_masks = True

    if rhoMaskFrac is not None:
      self.rhoMaskFrac = rhoMaskFrac
      recalc_masks = True
    if (self.rhoMaskFrac is not None):
      recalc_masks = True

    if (self.tSigma is None) and (self.dTheta is not None):
      self.tSigma = 1.0/self.dTheta
      recalc_masks = True

    if (self.rSigma is None) and (self.dRho is not None):
      self.rSigma = 0.25/np.float32(self.dRho)
      recalc_masks = True

    if recalc_masks == True:
      ksz = np.array([np.max([np.int64(4*self.rSigma), 5]), np.max([np.int64(4*self.tSigma), 5])])
      ksz = ksz + ((ksz % 2) == 0)
      kernel = np.zeros(ksz, dtype=np.float32)
      kernel[(ksz[0]/2).astype(int),(ksz[1]/2).astype(int) ] = 1
      kernel = -1.0*gaussian_filter(kernel, [self.rSigma, self.tSigma], order=[2,0])
      self.kernel = kernel.reshape((1,ksz[0], ksz[1]))
      #self.peakPad = np.array(np.around([ 4*ksz[0], 20.0/self.dTheta]), dtype=np.int64)
      self.peakPad = np.array(np.around([3 * ksz[0], 4 * ksz[1]]), dtype=np.int64)
      self.peakPad += 1 - np.mod(self.peakPad, 2)  # make sure we have it as odd.

    self.padding = np.array([np.max( [self.peakPad[0], self.padding[0]] ), np.max([self.peakPad[1], self.padding[1]])])

    if nBands is not None:
      self.nBands = nBands

  def collect_background(self, fileobj = None, patsIn = None, nsample = None, method = 'randomStride', sigma=None):

    back = None # default value
    # we got an array of patterns

    if patsIn is not None:
      ndim = patsIn.ndim
      if ndim == 2:
        patsIn = np.expand_dims(patsIn,axis=0)
      else:
        patsIn = patsIn
      npats = patsIn.shape[0]
      if nsample is None:
        nsample = npats
      #pshape = patsIn.shape
      if npats <= nsample:
        back = np.mean(patsIn, axis = 0)
        back = np.expand_dims(back,axis=0)
      else:
        if method.upper() == 'RANDOMSTRIDE':
          stride = np.random.choice(npats, size = nsample, replace = False )
          stride = np.sort(stride)
          back = np.mean(patsIn[stride,:,:],axis=0)
        elif method.upper() == 'EVENSTRIDE':
          stride = np.arange(0, npats, int(npats/nsample)) # not great, but maybe good enough.
          back = np.mean(patsIn[stride, :, :], axis=0)

    if (back is None) and (fileobj is not None):
      if fileobj.version is None:
        fileobj.read_header()
      npats = fileobj.nPatterns
      if nsample is None:
        nsample = npats
      if npats <= nsample:
        nsample = npats

      if method.upper() == 'RANDOMSTRIDE':
        stride = np.random.choice(npats, size = nsample, replace = False )
        stride = np.sort(stride)
      elif method.upper() == 'EVENSTRIDE':
        step = int(npats / nsample) # not great, but maybe good enough.
        stride = np.arange(0,npats, step, dypte = np.uint64)
      pat1 = fileobj.read_data(convertToFloat=True,patStartCount=[stride[0], 1],returnArrayOnly=True)
      for i in stride[1:]:
        pat1 += fileobj.read_data(convertToFloat=True,patStartCount=[i, 1],returnArrayOnly=True)
      back = pat1 / float(len(stride))
      #pshape = pat1.shape
    # a bit of image processing.
    if back is not None:
      #if sigma is None:
       #sigma = 2.0 * float(pshape[-1]) / 80.0
      #back[0,:,:] = gaussian_filter(back[0,:,:], sigma = sigma )
      back -= np.mean(back)
    self.backgroundsub = back

  def find_bands(self, patternsIn, verbose=0, chunksize=-1,  **kwargs):
    tic0 = timer()
    tic = timer()
    ndim = patternsIn.ndim
    if ndim == 2:
      patterns = np.expand_dims(patternsIn, axis=0)
    else:
      patterns = patternsIn

    shape = patterns.shape
    nPats = shape[0]

    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)
    if chunksize < 0:
      nchunks = 1
      chunksize = nPats
      chunk_start_end = [[0,nPats]]
    else:
      nchunks = (np.ceil(nPats / chunksize)).astype(np.compat.long)
      chunk_start_end = [[i * chunksize, (i + 1) * chunksize] for i in range(nchunks)]
      chunk_start_end[-1][1] = nPats

    # these are timers used to gauge performance
    rdntime = 0.0
    convtime = 0.0
    lmaxtime = 0.0
    blabeltime = 0.0

    for chnk in chunk_start_end:
      tic1 = timer()
      rdnNorm = self.radonPlan.radon_faster(patterns[chnk[0]:chnk[1],:,:], self.padding, fixArtifacts=True, background=self.backgroundsub)
      rdntime += timer() - tic1
      tic1 = timer()
      rdnConv = self.rdn_conv(rdnNorm)
      convtime += timer()-tic1
      tic1 = timer()
      lMaxRdn= self.rdn_local_max(rdnConv)
      lmaxtime +=  timer()-tic1
      tic1 = timer()
      bandDataChunk= self.band_label(chnk[1]-chnk[0], rdnConv, rdnNorm, lMaxRdn)
      bandData[chnk[0]:chnk[1]] = bandDataChunk
      if (verbose > 1) and (chnk[1] == nPats): # need to pull the radonconv off the gpu
        rdnConv = rdnConv[:,:,0:chnk[1]-chnk[0] ]

      blabeltime += timer() - tic1

    tottime = timer() - tic0

    if verbose > 0:
      print('Radon Time:',rdntime)
      print('Convolution Time:', convtime)
      print('Peak ID Time:', lmaxtime)
      print('Band Label Time:', blabeltime)
      print('Total Band Find Time:',tottime)
    if verbose > 1:
      plt.clf()

      if len(rdnConv.shape) == 3:
        im2show = rdnConv[self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1], -1]
      else:
        im2show = rdnConv[self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]

      rhoMaskTrim = np.int32(im2show.shape[0] * self.rhoMaskFrac)
      mean = np.mean(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])
      stdv = np.std(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])

      #im2show -= mean
      #im2show /= stdv
      #im2show += 7
      im2show[0:rhoMaskTrim,:] = 0
      im2show[-rhoMaskTrim:,:] = 0

      plt.imshow(im2show, origin='lower', cmap='gray')
      #plt.scatter(y = bandData['aveloc'][-1,:,0], x = bandData['aveloc'][-1,:,1]+self.peakPad[1], c ='r', s=5)
      plt.scatter(y=bandData['aveloc'][-1,:,0],x=bandData['aveloc'][-1,:,1],c='r',s=5)
      plt.xlim(0,self.nTheta)
      plt.ylim(0,self.nRho)
      plt.show()


    return bandData

  def radonPad(self,radon,rPad=0,tPad = 0, mirrorTheta = True):
    # function for padding the radon transform
    # theta padding (tPad) will use the symmetry of the radon and will vertical flip the transform and place it on
    # the other side.
    # rho padding simply repeats the top/bottom rows into the padded region
    # negative padding values will result in a crop (remove the padding).
    shp = radon.shape
    if (tPad==0)&(rPad == 0):
      return radon


    if (mirrorTheta == True)&(tPad > 0):
      rdnP = np.zeros((shp[0],shp[1],shp[2] + 2 * tPad),dtype=np.float32)
      rdnP[:,:,tPad:-tPad] = radon
        # now pad out the theta dimension with the flipped-wrapped radon -- accounting for radon periodicity
      rdnP[:,:,0:tPad] = np.flip(rdnP[:,:,-2 *tPad:-tPad],axis=1)
      rdnP[:,:,-tPad:] = np.flip(rdnP[:,:,tPad:2 * tPad],axis=1)
    else:
      if tPad > 0:
        rdnP = np.pad(radon,((0,),(0,),(tPad,)),mode='edge')
      elif tPad < 0:
        rdnP = radon[:,:,-tPad:tPad]
      else:
        rdnP = radon

    if rPad > 0:
      rdnP =  np.pad(rdnP,((0,),(rPad,),(0,)),mode='edge')
    elif rPad < 0:
      rdnP = rdnP[:,-rPad:rPad, :]

    return rdnP



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
    theta = np.pi - self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1],dtype=np.int64)] / RADEG
    rho = -1.0*self.radonPlan.rho[np.array(bandData['aveloc'][:,:,0],dtype=np.int64)]

    # from this point on, we will assume the image origin and t-vector (aka pattern center) is described
    # at the bottom left of the pattern
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    t = np.asfarray(PC).copy()
    shapet = t.shape
    if len(shapet) < 2:
      t = np.tile(t, nPats).reshape(nPats,3)
    else:
      if shapet[0] != nPats:
        t = np.tile(t[0,:], nPats).reshape(nPats,3)

    dimf = np.array(self.patDim, dtype=np.float32)
    if ven in ['EDAX', 'OXFORD']:
      t *= np.array([dimf[1], dimf[1], -dimf[1]])
    if ven == 'EMSOFT':
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


  def rdn_conv(self, radonIn):
    tic = timer()
    shp = radonIn.shape
    if len(shp) == 2:
      radon = radonIn.reshape(shp[0],shp[1],1)
      shp = radon.shape
    else:
      radon = radonIn
    shprdn = radon.shape
    #rdnNormP = self.radonPad(radon,rPad=0,tPad=self.peakPad[1],mirrorTheta=True)
    if self.padding[1] > 0:
      radon[:,0:self.padding[1],:] = np.flip(radon[:,-2 * self.padding[1]:-self.padding[1],:],axis=0)
      radon[:,-self.padding[1]:,:] = np.flip(radon[:,self.padding[1]:2 * self.padding[1],:],axis=0)
    # pad again for doing the convolution
    #rdnNormP = self.radonPad(rdnNormP,rPad=self.peakPad[0],tPad=self.peakPad[1],mirrorTheta=False)
    if self.padding[0] > 0:
      radon[0:self.padding[0], :,:] = radon[self.padding[0],:,:].reshape(1,shp[1], shp[2])
      radon[-self.padding[0]:, :,:] = radon[-self.padding[0]-1, :,:].reshape(1, shp[1],shp[2])


    rdnConv = np.zeros_like(radon)

    for i in range(shp[2]):
      rdnConv[:,:,i] = -1.0 * gaussian_filter(np.squeeze(radon[:,:,i]),[self.rSigma,self.tSigma],order=[2,0])

    #print(rdnConv.min(),rdnConv.max())
    mns = (rdnConv[self.padding[0]:shprdn[1]-self.padding[0],self.padding[1]:shprdn[1]-self.padding[1],:]).min(axis=0).min(axis=0)

    rdnConv -= mns.reshape((1,1, shp[2]))
    rdnConv = rdnConv.clip(min=0.0)

    return rdnConv

  def rdn_local_max(self, rdn, clparams=None, rdn_gpu=None, use_gpu=False):

    shp = rdn.shape
    # find the local max
    lMaxK = (self.peakPad[0],self.peakPad[1],1)

    lMaxRdn = scipy_grey_dilation(rdn,size=lMaxK)
    #lMaxRdn[:,:,0:self.peakPad[1]] = 0
    #lMaxRdn[:,:,-self.peakPad[1]:] = 0
    #location of the max is where the local max is equal to the original.
    lMaxRdn = lMaxRdn == rdn

    rhoMaskTrim = np.int32((shp[0] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])
    lMaxRdn[0:rhoMaskTrim,:,:] = 0
    lMaxRdn[-rhoMaskTrim:,:,:] = 0
    lMaxRdn[:,0:self.padding[1],:] = 0
    lMaxRdn[:,-self.padding[1]:,:] = 0
    #print("Traditional:",timer() - tic)
    return lMaxRdn


  def band_label(self,nPats,rdnConvIn,rdnNormIn,lMaxRdnIn):
    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)


    bdat = self.band_label_numba(
      np.int64(self.nBands),
      np.int64(nPats),
      np.int64(self.nRho),
      np.int64(self.nTheta),
      rdnConvIn,
      rdnConvIn,
      lMaxRdnIn
    )

    bandData['max']    = bdat[0][0:nPats, :]
    bandData['avemax'] = bdat[1][0:nPats, :]
    bandData['maxloc'] = bdat[2][0:nPats, :, :]
    bandData['aveloc'] = bdat[3][0:nPats, :, :]
    bandData['valid']  = bdat[4][0:nPats, :]
    bandData['width']  = bdat[5][0:nPats, :]
    bandData['maxloc'] -= self.padding.reshape(1, 1, 2)
    bandData['aveloc'] -= self.padding.reshape(1, 1, 2)

    return bandData

  @staticmethod
  @numba.jit(nopython=True,fastmath=True,cache=True,parallel=False)
  def band_label_numba(nBands,nPats,nRho,nTheta,rdnConv,rdnPad,lMaxRdn):
    nB = np.int64(nBands)
    nP = np.int64(nPats)

    shp = rdnPad.shape

    bandData_max = np.zeros((nP,nB),dtype=np.float32) - 2.0e6  # max of the convolved peak value
    bandData_avemax = np.zeros((nP,nB),
                               dtype=np.float32) - 2.0e6  # mean of the nearest neighborhood values around the max
    bandData_valid = np.zeros((nP, nB), dtype=np.int8)
    bandData_maxloc = np.zeros((nP,nB,2),dtype=np.float32)  # location of the max within the radon transform
    bandData_aveloc = np.zeros((nP,nB,2),
                               dtype=np.float32)  # location of the max based on the nearest neighbor interpolation
    bandData_width = np.zeros((nP,nB),dtype=np.float32) # a metric of the band width

    nnc = np.array([-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2],dtype=np.float32)
    nnr = np.array([-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1],dtype=np.float32)
    nnN = numba.float32(15)

    for q in range(nPats):
      # rdnConv_q = np.copy(rdnConv[:,:,q])
      # rdnPad_q = np.copy(rdnPad[:,:,q])
      # lMaxRdn_q = np.copy(lMaxRdn[:,:,q])
      # peakLoc = np.nonzero((lMaxRdn_q == rdnPad_q) & (rdnPad_q > 1.0e-6))
      peakLoc = lMaxRdn[:,:,q].nonzero()
      indx1D = peakLoc[1] + peakLoc[0] * shp[1]
      temp = (rdnConv[:,:,q].ravel())[indx1D]
      srt = np.argsort(temp)
      nBq = nB if (len(srt) > nB) else len(srt)
      for i in numba.prange(nBq):
        r = np.int32(peakLoc[0][srt[-1 - i]])
        c = np.int32(peakLoc[1][srt[-1 - i]])
        bandData_maxloc[q,i,:] = np.array([r,c])
        bandData_max[q,i] = rdnPad[r,c,q]
        bandData_width[q, i] = 1.0 / (bandData_max[q,i] - 0.5* (rdnPad[r+1, c, q] + rdnPad[r-1, c, q]) + 1.0e-12)
        # nn = rdnPad_q[r - 1:r + 2,c - 2:c + 3].ravel()
        nn = rdnConv[r - 1:r + 2,c - 2:c + 3,q].ravel()
        sumnn = (np.sum(nn) + 1.e-12)
        nn /= sumnn
        bandData_avemax[q,i] = sumnn / nnN
        rnn = np.sum(nn * (np.float32(r) + nnr))
        cnn = np.sum(nn * (np.float32(c) + nnc))
        bandData_aveloc[q,i,:] = np.array([rnn,cnn])
        bandData_valid[q,i] = 1
    return bandData_max,bandData_avemax,bandData_maxloc,bandData_aveloc, bandData_valid, bandData_width

def getopenclparam(): # dummy function to maintain compatability with openCL version
  return None

