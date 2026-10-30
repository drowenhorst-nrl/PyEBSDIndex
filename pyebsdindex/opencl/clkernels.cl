/*
This software was developed by employees of the US Naval Research Laboratory (NRL), an
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
The US Naval Research Laboratory Date: 05 Aug 2026

 File: clkernels.cl
 --------------------------------- 
 This file contains OpenCL kernels for PyEBSDIndex, a library for Electron Backscatter 
 Diffraction (EBSD) pattern processing and analysis. The kernels implement various image
 processing operations including:
 
 - Data loading and conversion (loadubyte8, loaduint16, loadfloat32)
 - Background subtraction and division (backSub, backDiv)
 - Radon transform operations (radonSum, radonFixArt, radonPadTheta, radonPadRho)
 - Image convolution and morphological operations (convolution3d2d, morphDilateKernelBF)
 - Image statistics and normalization (imageMinAveMax, imageSubMinNormWClip)
 - Peak detection and analysis (maxlabel, dirtsort)
 
 These kernels are designed to run on GPUs and other OpenCL-compatible devices to
 accelerate EBSD pattern processing tasks.
*/

// simple program to convert a 8-bit byte to float and transpose array
// This kernel is used in the image loading pipeline to convert unsigned char (8-bit) 
// image data to float format with array transposition for efficient memory access
// patterns in subsequent processing steps.
// 
// Parameters:
//   im1:    Input 8-bit image data (unsigned char array)
//   im1flt: Output float image data
//   nImCL:  Number of images to process in this chunk
//
// The transposition changes the memory layout from [x,y,z] to [z,x,y] for better
// GPU memory access performance in subsequent operations
__kernel void loadubyte8( const __global uchar *im1, __global float *im1flt, const unsigned long int nImCL)
  {
  const unsigned long int x = get_global_id(0);
  const unsigned long int y = get_global_id(1);
  const unsigned long int z = get_global_id(2);
  const unsigned long int nx = get_global_size(0);
  const unsigned long int ny = get_global_size(1);
  //const unsigned long int nz = get_global_size(2);
  
  const unsigned long int indx1 = x + nx*y + nx*ny*z;
  const unsigned long int indx2 = z + nImCL*(x + nx*y); // transpose z->x, x->y, y->z 
  uchar imVal;
  float imValflt;
  
  imVal =  im1[indx1];
  imValflt = convert_float(imVal);
  im1flt[indx2] = imValflt;
  
  
}


// simple program to convert a 16-bit unsigned short to float and transpose array
// This kernel is used in the image loading pipeline to convert unsigned short (16-bit) 
// image data to float format with array transposition for efficient memory access
// patterns in subsequent processing steps.
// 
// Parameters:
//   im1:    Input 16-bit image data (unsigned short array) - processes images individually
//   im1flt: Output float image data - processes images individually  
//   nImCL:  Number of images to process in this chunk - typically processes single images
//
// The transposition changes the memory layout from [x,y,z] to [z,x,y] for better
// GPU memory access performance in subsequent operations
__kernel void loaduint16( const __global ushort *im1, __global float *im1flt, const unsigned long int nImCL)
  {
  const unsigned long int x = get_global_id(0);
  const unsigned long int y = get_global_id(1);
  const unsigned long int z = get_global_id(2);
  const unsigned long int nx = get_global_size(0);
  const unsigned long int ny = get_global_size(1);
  //const unsigned long int nz = get_global_size(2);
  
  const unsigned long int indx1 = x + nx*y + nx*ny*z;
  const unsigned long int indx2 = z + nImCL*(x + nx*y); // transpose z->x, x->y, y->z 
  ushort imVal;
  float imValflt;

  imVal =  im1[indx1];
  imValflt = convert_float(imVal);
  im1flt[indx2] = imValflt;
  
  
}



// simple program to convert a 32-bit float to float and transpose array
// This kernel is used in the image loading pipeline to convert float (32-bit) 
// image data to float format with array transposition for efficient memory access
// patterns in subsequent processing steps.
// 
// Parameters:
//   im1:    Input 32-bit float image data (float array) - processes images individually
//   im1flt: Output float image data - processes images individually
//   nImCL:  Number of images to process in this chunk - typically processes single images
//
// The transposition changes the memory layout from [x,y,z] to [z,x,y] for better
// GPU memory access performance in subsequent operations
__kernel void loadfloat32( const __global float *im1, __global float *im1flt, const unsigned long int nImCL)
  {
  const unsigned long int x = get_global_id(0);
  const unsigned long int y = get_global_id(1);
  const unsigned long int z = get_global_id(2);
  const unsigned long int nx = get_global_size(0);
  const unsigned long int ny = get_global_size(1);
  //const unsigned long int nz = get_global_size(2);
  
  const unsigned long int indx1 = x + nx*y + nx*ny*z;
  const unsigned long int indx2 = z + nImCL*(x + nx*y); // transpose z->x, x->y, y->z 
  float imVal;

  imVal =  im1[indx1];
  im1flt[indx2] = imVal;
  
  
}


// Perform background subtraction on EBSD patterns
// This kernel subtracts a pre-computed background pattern from each image in a stack
// to remove background noise and improve pattern visibility.
// 
// Parameters:
//   im1:         Input/Output image stack (float16 array) - modified in-place
//                Processes images in batches of nImChunk (typically 16 images at once)
//   back:        Background pattern to subtract (float array) - single background for all images
//   nImChunk:    Number of images in the current processing batch (typically 16)
//
// Each pixel in the image stack has the corresponding background value subtracted from it.
// The background pattern is typically computed as an average or median of many patterns.
// This operation processes multiple images in parallel for efficiency.
__kernel void backSub( __global float16 *im1, __global const float *back,
                        const unsigned long int nImChunk)
  {
  const unsigned long int xy = get_global_id(0);
  //const unsigned long int szim = get_global_size(0);
  unsigned long i;
  float16 imVal;

  const float b1 = back[xy];

  const unsigned long indx = nImChunk * xy;
  for(i = 0; i< nImChunk; ++i){
    imVal =  im1[indx+i];
    imVal -= b1;   
    im1[indx+i] = imVal;
  }
  
}

// Perform background division on EBSD patterns
// This kernel divides each image in a stack by a pre-computed background pattern
// to normalize intensity and remove background variations.
// 
// Parameters:
//   im1:         Input/Output image stack (float16 array) - modified in-place
//   back:        Background pattern to divide by (float array)
//   nImChunk:    Number of images in the current processing chunk
//
// Each pixel in the image stack is divided by the corresponding background value.
// Background values less than 1.0 are clamped to 1.0 to avoid division by zero or amplification.
// This operation helps normalize patterns and reduce the effects of uneven illumination.
__kernel void backDiv( __global float16 *im1, __global const float *back,
                        const unsigned long int nImChunk)
  {
  const unsigned long int xy = get_global_id(0);
  //const unsigned long int szim = get_global_size(0);
  unsigned long i;
  float16 imVal;

  float b1 = back[xy];
  if (b1 < 1.0){
    b1 = 1.0;
  }
  
  const unsigned long indx = nImChunk * xy;
  for(i = 0; i< nImChunk; ++i){
    imVal =  im1[indx+i];
    imVal /= b1;   
    im1[indx+i] = imVal;
  }
  
}


// Compute Radon transform of EBSD patterns
// This kernel computes the Radon transform (projection) of EBSD patterns along specified
// lines in the pattern. The Radon transform is used for detecting bands in EBSD patterns.
// 
// Parameters:
//   rdnIndx:   Array of indices for Radon transform lines (unsigned long int array)
//   images:    Input EBSD pattern images (float16 array)
//   radon:     Output Radon transform results (float16 array)
//   imstep:    Size of one image in pixels
//   indxstep:  Number of indices per Radon line
//   nRhoP:     Number of rho (distance) values in padded Radon space
//   nThetaP:   Number of theta (angle) values in padded Radon space
//   rhoPad:    Padding size for rho dimension
//   thetaPad:  Padding size for theta dimension
//   nTheta:    Number of theta values in original Radon space
//
// The kernel processes each image in parallel, computing the average intensity along each
// specified line in the Radon transform space. Missing values are indicated by -1.0.
__kernel void radonSum(
      __global const unsigned long int *rdnIndx, __global const float16 *images, __global float16 *radon,
      const unsigned long int imstep, const unsigned long int indxstep,
      const unsigned int long nRhoP, const unsigned long int nThetaP,
      const unsigned long int rhoPad, const unsigned long int thetaPad,const unsigned long int nTheta )
    {

    float16 sum = (float16)(0.0);
    float count = 1.0e-12;
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int q = get_global_id(1);
    const unsigned long int rho = q/nTheta;
    const unsigned long int theta = q % nTheta;

    const unsigned long int indxStart = (theta + nTheta * rho)*indxstep;
    unsigned long int i, idx;

    for (i=0; i<indxstep; i++){
      idx = rdnIndx[indxStart+i];
      if (idx >= imstep) {
        break;
      } else {
        sum += images[idx*nImChunk + gid_im];
        count += 1.0;
      }
    }


    const unsigned long int rndIndx = (theta+thetaPad + (rho+rhoPad)*nThetaP)*nImChunk + gid_im;
    if (count > 1.0e-6){
      radon[rndIndx] = sum/count;}
    else{
      radon[rndIndx] = -1.0;
    }
    
    }

// Fix artifacts in Radon transform at theta boundaries
// This kernel corrects artifacts that occur at the theta=0 and theta=180 degree boundaries
// of the Radon transform by copying values from adjacent valid positions.
// 
// Parameters:
//   radon:      Input/Output Radon transform data (float16 array) - modified in-place
//   nRho:       Number of rho (distance) values
//   nTheta:     Number of theta (angle) values
//   thetaPad:   Padding size for theta dimension
//
// The kernel copies values from theta=1 to theta=0 and from theta=nTheta-2 to theta=nTheta-1
// to fix boundary artifacts that can occur due to the periodic nature of the Radon transform.
__kernel void radonFixArt(
      __global float16 *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int thetaPad)
  {
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int rho = get_global_id(1);
    const unsigned long int rhoIndx = nTheta * rho;
    //rndIndx = nTheta * nRho * gid_im + (nTheta * rho);
    
    radon[(thetaPad + rhoIndx)*nImChunk + gid_im] = radon[(thetaPad + 1 + rhoIndx)*nImChunk + gid_im];
   
    radon[(nTheta-1-thetaPad + rhoIndx)*nImChunk + gid_im] = radon[(+nTheta-2-thetaPad + rhoIndx)*nImChunk + gid_im];
    //}
  }


// // Makes a mask based on if the Radon has missing values -- returns byte array.
// // There is NOT ability to include padding in x and y ... yet.  
// __kernel void maskrdn( __global const float *im1, __global uchar *out, const float misval)
// {
//   // IDs of work-item represent x and y coordinates in image
//   //const unsigned long int x = get_global_id(0)+padx;
//   //const unsigned long int y = get_global_id(1)+pady;
//   const unsigned long x = get_global_id(0);
//   const unsigned long int imszx = get_global_size(0);
//   const unsigned long y = get_global_id(1);
//   const unsigned long int z = get_global_id(2);
//   const unsigned long int nImChunk = get_global_size(2);
//   //const int16 yes = (int16)(1);
//   //const int16 no = (int16)(0);

//   const unsigned long int indx = (x + imszx*y)*nImChunk + z;
//   const float im1val = im1[indx];
  
//   const float diff = fabs(im1val - misval);
//   //out[indx] = (diff < 1.0e-6f) ? yes: no;
//   out[indx] = (diff < 1.0e-5f) ? 0: 1;
// }



// Pad Radon transform in theta dimension
// This kernel performs symmetric padding in the theta dimension of the Radon transform
// by copying values from the opposite side of the transform space.
// 
// Parameters:
//   radon:      Input/Output Radon transform data (float array) - modified in-place
//   nRho:       Number of rho (distance) values
//   nTheta:     Number of theta (angle) values
//   thetaPad:   Number of padding elements to add to theta dimension
//
// The padding uses the symmetry property of the Radon transform where theta and theta+180
// are related by vertical flipping. This creates symmetric padding that helps with
// subsequent processing steps like convolution.
__kernel void radonPadTheta(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int thetaPad)
  {
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int gid_rho = get_global_id(1);
    unsigned long int gid_rdn1,gid_rdn2, i;
    gid_rdn1 =   (nTheta * gid_rho);
    gid_rdn2 =   (nTheta * (nRho -1 - gid_rho));

    for (i = 0; i <= thetaPad; ++i){
        radon[(i + gid_rdn1)*nImChunk + gid_im] = radon[ (gid_rdn2 + (nTheta-1 - (2 * thetaPad)) + i) * nImChunk + gid_im ];
        radon[ (gid_rdn2 + (nTheta-1 - ( thetaPad)) + i) * nImChunk + gid_im] = radon[ (gid_rdn1 + thetaPad + i)*nImChunk +gid_im];
    }

  }

// Pad Radon transform in rho dimension
// This kernel performs padding in the rho dimension of the Radon transform
// by copying values from the nearest valid line.
// 
// Parameters:
//   radon:      Input/Output Radon transform data (float array) - modified in-place
//   nRho:       Number of rho (distance) values
//   nTheta:     Number of theta (angle) values
//   rhoPad:     Number of padding elements to add to rho dimension
//
// The padding copies the values from the first and last valid rho lines to create
// padding at the boundaries. This helps with subsequent processing steps and prevents
// edge artifacts in operations like convolution.
__kernel void radonPadRho(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int rhoPad)
  {
    const unsigned long int gid_im = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int gid_theta = get_global_id(1);
    unsigned long int i, gid_rdn1, gid_rdn2;
    //indxim = nTheta * nRho * gid_im;
    //rd1p =  radon[indxim + (nTheta * rhoPad) + gid_theta] ;
    //rd2p =  radon[ indxim + (nTheta * (nRho -1 - rhoPad)) + gid_theta] ;
    const float rd1p =  radon[((nTheta * rhoPad) + gid_theta)*nImChunk + gid_im] ;
    const float rd2p =  radon[((nTheta * (nRho -1 - rhoPad)) + gid_theta)*nImChunk+gid_im];

    for (i = 0; i <= rhoPad; ++i){

        //gid_rdn1 = indxim + (nTheta*i) + gid_theta;
        //gid_rdn2 = indxim + (nTheta* (nRho-1-rhoPad+i)) + gid_theta;
        gid_rdn1 = ((nTheta*i) + gid_theta)*nImChunk + gid_im;
        gid_rdn2 = ((nTheta* (nRho-1-rhoPad+i)) + gid_theta)*nImChunk + gid_im;
        radon[gid_rdn1] = rd1p;
        radon[gid_rdn2] = rd2p;
    }

  }


  // Padding of the radon Rho -- copy the previous line to the next row ...
  __kernel void radonPadRho2(
      __global float *radon,
      const unsigned long int nRho, const unsigned long int nTheta,
      const unsigned long int rhoPad)
  {
    const unsigned long int z = get_global_id(0);
    const unsigned long int nImChunk = get_global_size(0);
    const unsigned long int gid_theta = get_global_id(1);
    unsigned long int i, gid_rdn1, gid_rdn2;
    //indxim = nTheta * nRho * z;
    //rd1p =  radon[indxim + (nTheta * rhoPad) + gid_theta] ;
    //rd2p =  radon[ indxim + (nTheta * (nRho -1 - rhoPad)) + gid_theta] ;
    
    gid_rdn1 = ((nTheta*rhoPad) + gid_theta)*nImChunk + z;
    gid_rdn2 = ((nTheta* (nRho-1-rhoPad)) + gid_theta)*nImChunk + z;

    float rd1p =  radon[gid_rdn1];
    float rd2p =  radon[gid_rdn2];
    

    for (i = 0; i <= rhoPad; ++i){

        //gid_rdn1 = indxim + (nTheta*i) + gid_theta;
        //gid_rdn2 = indxim + (nTheta* (nRho-1-rhoPad+i)) + gid_theta;
        
        gid_rdn1 = ( (nTheta* (rhoPad - i)) + gid_theta)*nImChunk + z;
        gid_rdn2 = ((nTheta* (nRho-1-rhoPad+i)) + gid_theta)*nImChunk + z;
        
        if (radon[gid_rdn1] < 0){
          radon[gid_rdn1] = rd1p;
        }

        if (radon[gid_rdn2] < 0){
          radon[gid_rdn2] = rd2p;
        }
        rd1p =  radon[gid_rdn1] ;
        rd2p =  radon[gid_rdn2] ;

    }

  }


// Perform 2D convolution on a stack of 3D images
// This kernel applies a 2D convolution kernel to each image in a 3D stack.
// The convolution is used for smoothing, edge detection, or other filtering operations.
// 
// Parameters:
//   in:     Input image stack (float16 array)
//   kern:   Convolution kernel (float array) - stored in constant memory
//   imszx:  Image width in pixels
//   imszy:  Image height in pixels
//   imszz:  Number of images in stack
//   kszx:   Kernel width in pixels
//   kszy:   Kernel height in pixels
//   padx:   Padding size in x dimension
//   pady:   Padding size in y dimension
//   out:    Output convolved image stack (float16 array)
//
// The kernel handles edge conditions by adjusting the convolution bounds and uses
// the full kernel for interior pixels. The output is computed as the sum of
// element-wise products between the kernel and the corresponding image region.
__kernel void convolution3d2d( __global const float16 *in, __constant float *kern, const int imszx, const int imszy, const int imszz, 
  const int kszx, const int kszy, const int padx, const int pady, __global float16 *out)
{
  // IDs of work-item represent x and y coordinates in image
  const long x = get_global_id(0) + padx;
  const long y = get_global_id(1) + pady;
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);
  //long int indxIm; 
  //long int indxK;
  unsigned long yIndx; 
  unsigned long yKIndx; 
  float16 pxVal;
  float16 sum = (float16) (0.0f); // initialize convolution sum
  float kVal;
  const int kszx2 = kszx/2;
  const int kszy2 = kszy/2;

  const long int istart = ((x+kszx2) >= imszx) ? x+kszx2+1-imszx: 0;
  const long int iend = ((x-kszx2) < 0) ? x+kszx2+1: kszx;
  const long int startx = x+kszx2;
  const long int jstart = ((y+kszy2) >= imszy) ? y+kszy2+1-imszy: 0;
  const long int jend = ((y-kszy2) < 0) ? y+kszy2+1: kszy;
  const long int starty = y+kszy2;
  
  
  
  //const long int zIndx = imszx*imszy*z;
  //float current = out[(y*imszx + x) * nImChunk + z];

  for(int j=jstart; j<jend; ++j)
  {
      //yIndx = zIndx + (j)*imszx;
      //yKIndx = (j-y)*kszx;
      yKIndx = kszx*(j);
      yIndx = imszx*(starty - j);
      for(int i=istart; i<iend; ++i)
      {      
        pxVal = in[(yIndx+(startx - i)) * nImChunk + z];
        kVal = kern[yKIndx + i];
        sum += pxVal * ((float16) kVal);
    }
  }
  // sum = current < sum ? sum : current;
  // write value of pixel to output image at location (x, y,z)
  out[(y*imszx + x) * nImChunk + z] = sum;
}

// Perform brute-force grayscale morphological dilation
// This kernel implements a local maximum operation (dilation) using a brute-force approach.
// Dilation expands bright regions and is used for image enhancement and feature detection.
// 
// Parameters:
//   in:     Input image stack (float16 array)
//   out:    Output dilated image stack (float16 array)
//   imszx:  Image width in pixels
//   imszy:  Image height in pixels
//   padx:   Padding size in x dimension
//   pady:   Padding size in y dimension
//   kszx:   Structuring element width in pixels
//   kszy:   Structuring element height in pixels
//
// For each pixel, the kernel finds the maximum value in the neighborhood defined by the
// structuring element (kernel size). Edge pixels use smaller neighborhoods to stay within
// image bounds. The result expands bright features and fills small dark holes.
//
// Note: This brute-force implementation may be replaced with more efficient algorithms
// like van Herk/Gil-Werman for better performance on large kernels.
__kernel void morphDilateKernelBF( __global const float16 *in,  __global float16 *out,
                                   const long imszx, const long imszy, 
                                   const long padx, const long pady,
                                   const long kszx, const long kszy)
{
  //IDs of work-item represent x and y coordinates in image; z is the n-th image.
  //const long int x = get_global_id(0) + xpad;
  //const long int y = get_global_id(1) + ypad;
  const long x = get_global_id(0) + padx;
  const long y = get_global_id(1) + pady;
  const long int z = get_global_id(2);
  const long int nImChunk = get_global_size(2);
  long int yIndx;
  float16 pxVal;
  const  long kszx2 = kszx/2;
  const  long kszy2 = kszy/2;

  const long int istart = x-kszx2 >= 0 ? x-kszx2: 0;
  const long int iend = x+kszx2 < imszx ? x+kszx2: imszx-1;
  const long int jstart = y-kszy2 >= 0 ? y-kszy2: 0;
  const long int jend = y+kszy2 < imszy ? y+kszy2: imszy-1;

  // extreme pixel value (max/min for dilation/erosion) inside structuring element
  float16 extremePxVal = (float16) (-1.0e12f); // initialize extreme value with min when searching for max value


  for(int j=jstart; j<=jend; ++j)
  {
      yIndx = j*imszx;

      for(int i=istart; i<=iend; ++i)
      {

      pxVal = in[(yIndx+i)*nImChunk+z];
      //const float pxVal = read_imagef(in, sampler_dilate, (int3)(x + i, y + j, z)).s0;
      extremePxVal = max(extremePxVal, pxVal);
    }
  }

  // write value of pixel to output image at location (x, y,z)
  out[(y*imszx + x)*nImChunk+z] = extremePxVal;
}

// Compute image statistics (minimum, average, maximum)
// This kernel calculates basic statistics for each image in a stack, which are used
// for normalization and other preprocessing steps.
// 
// Parameters:
//   im1:     Input image stack (float16 array)
//   imMin:   Output minimum intensity values (float16 array) - one per image
//   imMax:   Output maximum intensity values (float16 array) - one per image
//   imAve:   Output average intensity values (float16 array) - one per image
//   imszx:   Image width in pixels
//   imszy:   Image height in pixels
//   padx:    Padding size in x dimension
//   pady:    Padding size in y dimension
//
// The kernel processes each image independently, computing the minimum, maximum, and
// average intensity values while excluding padding regions. These statistics are
// essential for subsequent normalization and contrast enhancement operations.
//
// Note: Performance could be improved by using work group reductions for parallel
// computation of statistics within each work group.
__kernel void imageMinAveMax( __global const float16 *im1, __global float16 *imMin, __global float16 *imMax, __global float16 *imAve,
                        const unsigned int imszx, const unsigned int imszy, const unsigned int padx, const unsigned int pady)
  {
  const unsigned long int z = get_global_id(0);
  const unsigned long int nImChunk = get_global_size(0);
  long int indx,i, j;
  float16 cmin = (float16) (1.0e12);
  float16 cmax = (float16) (-1.0e12);
  float16 cave = (float16) (0.0); 
  float16 imVal;


  //indxz = z*imszx*imszy;
  for(j = pady; j<= imszy - pady-1; ++j){
    indx = j*imszx;
    for(i = padx; i<= imszx - padx-1; ++i){
        imVal = im1[(indx+i)*nImChunk+z];
        cmin = select(cmin, imVal, (imVal < cmin));
        cmax = select(cmax, imVal, (imVal > cmax));
        cave += imVal; 
    }   
  }

  cave /= (float16) ( (imszy - 2*pady) * (imszx - 2*padx)); 
  //cave -= cmin;
  //cave = select(cave, (float16) (1.0f), (cave < (float16) (1.0e-8f)) );
  imAve[z] = cave; 
  imMin[z] = cmin;
  imMax[z] = cmax;
}

// Normalize images by subtracting minimum and clipping
// This kernel performs image normalization by subtracting the minimum intensity value
// from each image and applying clipping to ensure non-negative values.
// 
// Parameters:
//   im1:     Input/Output image stack (float16 array) - modified in-place
//   imMin:   Minimum intensity values (float16 array) - one per image
//   imAve:   Average intensity values (float16 array) - one per image (currently unused)
//   imszx:   Image width in pixels
//   imszy:   Image height in pixels
//   padx:    Padding size in x dimension
//   pady:    Padding size in y dimension
//
// For each pixel in each image, the kernel subtracts the image's minimum intensity value
// and clips the result to be non-negative. This operation helps normalize the dynamic range
// and prepare images for subsequent processing steps like peak detection.
//
// Note: The average intensity values (imAve) are passed but currently not used in this kernel.
__kernel void imageSubMinNormWClip( __global float16 *im1, 
  __global const float16 *imMin, __global const float16 *imAve, 
  const unsigned int imszx, const unsigned int imszy, 
  const unsigned int padx, const unsigned int pady)
  {
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long int x = get_global_id(0) + padx;
  //const unsigned long int y = get_global_id(1) + pady;
  const long x = get_global_id(0) + padx;
  const long y = get_global_id(1) + pady;
  const unsigned long z = get_global_id(2);
  const unsigned long nImChunk = get_global_size(2);

  const long int indx = (x+y*imszx)*nImChunk + z;
  const float16 im1val = im1[indx];
  float16 value = im1val - imMin[z];
  value = select(value, (float16) (0.0f), (value < (float16) (0.0f)) );
  //value *= ((float16) (1.0))/imAve[z];

  //im1[indx] = (value < (float16) (0.0f)) ? (float16) (0.0f) : value;
  im1[indx] = value; 
}



// Is image1 (can be a stack of images) EQ to image2 (can be a stack) -- returns byte array.
// There is ability to include padding in x and y
__kernel void im1EQim2( __global const float *im1, __global const float *im2, __global uchar *out,
                        const unsigned long imszx, const unsigned long imszy, 
                        const unsigned long padx, const unsigned long pady)
{
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long int x = get_global_id(0)+padx;
  //const unsigned long int y = get_global_id(1)+pady;
  const unsigned long x = get_global_id(0) + padx;
  const unsigned long y = get_global_id(1) + pady;
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);
  //const int16 yes = (int16)(1);
  //const int16 no = (int16)(0);

  const unsigned long int indx = (x + imszx*y)*nImChunk + z;
  const float im1val = im1[indx];
  const float im2val = im2[indx];
  const float diff = fabs(im1val - im2val);
  //out[indx] = (diff < 1.0e-6f) ? yes: no;
  out[indx] = (diff < 1.0e-5f) ? 1: 0;
  
  
}

// Is image1 (can be a stack of images) EQ to image2 (can be a stack) -- returns byte array.
// There is ability to include padding in x and y
__kernel void maskrdn( __global float16 *im1, __global const uchar *mask,
                        const unsigned long imszx, const unsigned long nImChunk,
                        const unsigned long padx, const unsigned long pady)
{
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long int x = get_global_id(0)+padx;
  //const unsigned long int y = get_global_id(1)+pady;
  const unsigned long x = get_global_id(0) + padx;
  const unsigned long y = get_global_id(1) + pady;
  unsigned long int indx_z,i;
  

  //const unsigned long int indx = (x + imszx*y)*nImChunk + z;
  const unsigned long int indx = (x + imszx*y);
  const uchar test = mask[indx];
  if (test < 1){
    for (i=0; i<nImChunk; ++i){
      indx_z = indx*nImChunk + i;
      im1[indx_z] = -1.0; // this should be an OK default value, cannot be a max
    }
    
  }
  
  
}


// Is image1 (can be a stack of images) EQ to image2 (can be a stack) -- returns byte array.
// There is ability to include padding in x and y
__kernel void maxmask( __global uchar16 *im1, __global const uchar *mask,
                        const unsigned long imszx, const unsigned long imszz, 
                        const unsigned long padx, const unsigned long pady)
{
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long int x = get_global_id(0)+padx;
  //const unsigned long int y = get_global_id(1)+pady;
  const unsigned long x = get_global_id(0) + padx;
  const unsigned long y = get_global_id(1) + pady;
  unsigned long int indx_z,i;
  

  //const unsigned long int indx = (x + imszx*y)*nImChunk + z;
  const unsigned long int indx = (x + imszx*y);
  const uchar test = mask[indx];
  if (test < 1){
    for (i=0; i<imszz; ++i){
      indx_z = indx*imszz + i;
      im1[indx_z] = 0;
    }
    
  }
  
  
}


// Simple insertion sort for peak detection
// This function performs a basic insertion sort to maintain a list of the top N maximum
// values and their locations. It's used during peak detection to keep track of the brightest
// peaks in EBSD patterns.
// 
// Parameters:
//   maxval1d:   Array of maximum values (float array) - maintains top N values in ascending order
//   maxloc1d:   Array of maximum locations (long int array) - corresponding locations
//   newloc:     Location of new candidate maximum
//   newmax:     Value of new candidate maximum
//   nmax:       Maximum number of peaks to track
//
// The function checks if the new value is larger than the smallest value currently in the
// top N list. If so, it inserts the new value in the correct position, shifting other
// values as needed. This maintains a sorted list of the top N peaks.
//
// Note: The array must be primed such that maxval1d[nmax-1] is larger than any value that
// could naturally occur, to ensure proper sorting behavior.
void dirtsort( __global float *maxval1d, __global long int *maxloc1d,
                long int newloc, float newmax, unsigned long nmax);

void dirtsort( __global float *maxval1d, __global long int *maxloc1d,
                long int newloc, float newmax, unsigned long nmax){

  unsigned long int i, istop;
  if (newmax < maxval1d[nmax-1]){
    return;
  }
  else{
    istop = 0; 
    for (i=1; i<nmax; ++i){
      if (newmax > maxval1d[nmax-1-i]){
        maxval1d[nmax-i] = maxval1d[nmax-1-i];
        maxloc1d[nmax-i] = maxloc1d[nmax-1-i];
        istop = i;
      }
      else{
        //maxval1d[i-1]  = newmax;
        //maxloc1d[i-1]  = newloc;
        break;
      }
    }
    maxval1d[nmax-1-istop] = newmax;
    maxloc1d[nmax-1-istop]  = newloc;
    return;
  }
}


// Detect and analyze peaks in EBSD patterns
// This kernel identifies local maxima (peaks) in EBSD patterns and performs sub-pixel
// interpolation to determine peak positions with higher precision than the original grid.
// 
// Parameters:
//   maxlocin:   Input peak location flags (uchar array) - 1 indicates potential peak
//   maxvalin:   Input intensity values (float array)
//   maxloc:     Output sorted peak locations (long int array) - [z*nmax + i]
//   maxval:     Output sorted peak values (float array) - [z*nmax + i]
//   aveloc:     Output sub-pixel peak positions (float2 array) - [z*nmax + i]
//   aveval:     Output interpolated peak values (float array) - [z*nmax + i]
//   width:      Output peak width estimates (float array) - [z*nmax + i]
//   imszx:      Image width in pixels
//   imszy:      Image height in pixels
//   padx:       Padding size in x dimension
//   pady:       Padding size in y dimension
//   nmax:       Maximum number of peaks to detect per image
//
// The kernel first identifies all local maxima using the input flags, then sorts them
// by intensity. For each significant peak, it performs sub-pixel interpolation using
// a Taylor series approximation to determine the peak position with higher precision.
// It also estimates peak widths assuming Gaussian profiles.
//
// The sub-pixel interpolation uses a 3x3 neighborhood around each peak to compute
// gradients and curvature, providing more accurate peak localization for subsequent
// crystallographic analysis.
__kernel void maxlabel( __global const uchar *maxlocin,__global const float *maxvalin,
                        __global long int *maxloc,__global float *maxval,
                        __global float2 *aveloc,__global float *aveval,
                        __global float *width,
                        const long imszx, const long imszy, 
                        const long padx, const long pady, const long nmax)
{
  // IDs of work-item represent x and y coordinates in image
  //const unsigned long x = get_global_id(0) + padx;
  //const unsigned long y = get_global_id(1) + pady;
  const unsigned long int z = get_global_id(0);
  const unsigned long int nImChunk = get_global_size(0);
  uchar imVal1;
  float imVal2, imValyp1, imValym1; 
  float dx, dy, dxx, dyy, dxy, det, w, ix, iy, a;
  const long lnmax = (long) nmax; 
  long int indy, indxy,i, j; //,k;
  //__global long int maxloc1d[32];// = {0};
  //__global float maxval1d[32];// = {-1e12f};
  long int x,y;
  float  avetempweight;
  float2  avetempxy;



  // first find all the local max points identified and sort them
  for (i=0;i<lnmax;++i){
    maxloc[i+lnmax*z] = 0;
    maxval[i+lnmax*z] = -1.0e12f;
  }

   
  //maxval1d[lnmax] = 1.0e12f; // prime the pump for sorting 
  for(j = pady; j< imszy - pady; ++j){
    indy = j*imszx;
    for(i = padx; i< imszx - padx; ++i){
        indxy = indy+i;

        imVal1 = maxlocin[(indxy)*nImChunk+z];
        //imVal2 = maxvalin[(indxy)*nImChunk+z];
        
        
        if (imVal1 == 0){
          continue;
        }
        else{
          imVal2 = maxvalin[(indxy)*nImChunk+z];
          
          dirtsort(&(maxval[z*lnmax]), &(maxloc[z*lnmax]), indxy, imVal2, lnmax);
        }
        
      
    }   
  }
  
  // now place them in the output arrays
  for (i=0; i< lnmax; ++i){
    if (maxval[z*lnmax+i] > 0.0){
    //if (maxval[z*lnmax+i] > -1.0e6){
      //maxval[z*lnmax + i] = maxval1d[lnmax-i-1];
      indxy = maxloc[z*lnmax+i];
      x =  ( indxy % imszx );
      y =  (indxy / imszx );
      //maxloc[z*lnmax + i] = (float2) (y,x);
      //maxloc[(z*lnmax + i)+1] = (float) y;
      avetempxy = (float2) (0.0);
      avetempweight = 1.0e-12;
      // // really basic 9-neighbor peak interpolation
      // for (k=-1; k<=1;++k){
      //   indy = (y+k)*imszx;
      //   for (j=-1; j<=1;++j){
      //     indxy = indy+(x+j);
      //     imVal2 = maxvalin[(indxy)*nImChunk+z];
      //     avetempxy += ((float2) (y+k, x+j))*imVal2;
      //     //avetempx += (x+j)*imVal2;
      //     avetempweight += imVal2;
      //   }
      // }
      // avetempxy /= avetempweight;
      // aveloc[z*lnmax + i] = avetempxy;
      // aveval[z*lnmax + i] = avetempweight/9.0;
      // // band width metric
      // imValyp1 = maxvalin[((y+1)*imszx+x)*nImChunk+z];
      // imValym1 = maxvalin[((y-1)*imszx+x)*nImChunk+z];
      // width[z*lnmax + i] = 1.0 / (maxval[z*lnmax+i] - 0.5 * (imValyp1 + imValym1) + 1e-12) ;

      // More advanced Taylor series approximation 
      dx = 0.0; dy = 0.0; dxx = 0.0; dyy = 0.0; dxy = 0.0; 
      indxy = ((y-1)*imszx + (x-1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy += imVal2; 
      
      //dx -= imVal2; 
      //dy -= imVal2;

      indxy = ((y-1)*imszx + (x))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dy  -= imVal2;
      //dy -= 2.0 * imVal2;
      dyy += imVal2;
      imValym1 = imVal2; 


      indxy = ((y-1)*imszx + (x+1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy -= imVal2;

      //dx += imVal2;
      //dy -= imVal2;

      indxy = ((y)*imszx + (x-1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dx -= imVal2; 
      //dx -= 2.0 * imVal2; 
      dxx += imVal2; 

      indxy = ((y)*imszx + (x))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxx -= 2.0 * imVal2;
      dyy -= 2.0 * imVal2;
      w = imVal2;

      indxy = ((y)*imszx + (x+1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dx  +=  imVal2;
      //dx += 2.0 * imVal2;
      dxx +=  imVal2;
      

      
      indxy = ((y+1)*imszx + (x-1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy -=  imVal2;

      //dx -= imVal2;
      //dy += imVal2;
      
      indxy = ((y+1)*imszx + (x))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dy  +=  imVal2;
      //dy += 2.0 * imVal2;
      dyy += imVal2; 
      imValyp1 = imVal2;

      indxy = ((y+1)*imszx + (x+1))*nImChunk+z; 
      imVal2 = maxvalin[indxy]; 
      avetempweight += imVal2;
      dxy  +=  imVal2;

      //dx += imVal2;
      //dy += imVal2; 

      dxy *= 0.25;
      dx  *= 0.5; //0.125; //
      dy  *= 0.5; //0.125; //

      det = (dxx*dyy - dxy*dxy);
      det = (fabs(det) > 1.0e-12) ? det : 1.0e-12; 
      det = 1.0 / det; 

      //ix = (float) x - (dyy * dx - dxy * dy) * det; 
      //iy = (float) y - (dxx * dy - dxy * dx) * det; 

      ix = (dyy * dx - dxy * dy) * det; 
      iy = (dxx * dy - dxy * dx) * det;
      // protect against bad estimates in dxy, assume == 0.0 -- per suggestion of W. Lenthe
      if ( (fabs(ix) > 0.875) || (fabs(iy) > 0.875) ){
        det = (dxx*dyy);
        det = (fabs(det) > 1.0e-12) ? det : 1.0e-12; 
        det = 1.0 / det; 
        ix = (dyy * dx) * det; 
        iy = (dxx * dy) * det;
        if ( (fabs(ix) > 0.875) || (fabs(iy) > 0.875) ){
          ix = 0.0; 
          iy = 0.0;
        }

      }

      ix = (float) x - ix;
      iy = (float) y - iy;
      
      aveloc[z*lnmax + i] = (float2) (iy, ix); 
      aveval[z*lnmax + i] = (avetempweight/9.0);
      // band width metric -- this older one assumes linear peak
      // width[z*lnmax + i] = 1.0 / (w - 0.5 * (imValyp1 + imValym1) + 1e-12) ;

      // this will assume a gaussian peak and provide FWHM
      a = (log(imValyp1) + log(imValym1)) * 0.5 - log(w);
      if (a < -1.e-8){
          width[z*lnmax + i] = 2.0 * sqrt(log(2.0) / (-1.0*a));
      } else{
        width[z*lnmax + i] = 0.0;
      }


    }
    else{
      break; // no more detected peaks
    }

  }
}














// Helper function for 1D morphological dilation
// This function performs a 1D dilation operation on a small window of data.
// It's used as a helper by the 2D dilation kernels to process rows or columns.
// 
// Parameters:
//   winsz:      Window size for dilation
//   winsz2:     Half window size (winsz/2)
//   s:          Scratch array for left-to-right pass
//   r:          Scratch array for right-to-left pass
//   w:          Working array containing the data to dilate
//
// The function uses a two-pass algorithm: first computing maximum values from
// left-to-right and right-to-left, then combining the results. This is more
// efficient than a brute-force approach for 1D dilation.
//
// Note: This function is marked as not fully working in the current implementation.
void morphDilateKernel(const unsigned long winsz, const unsigned long winsz2,
                      __local float16 *s,  __local float16 *r,  __local float16 *w);

void morphDilateKernel(const unsigned long winsz, const unsigned long winsz2,
                      __local float16 *s,  __local float16 *r,  __local float16 *w){
  unsigned long int i;
  const unsigned long  winsz1 = winsz-1;

  s[winsz1] = w[winsz1];
  r[0] = w[winsz1];
  for (i = 1; i< winsz; i++){
    s[winsz1-i] = max(s[winsz1-(i-1)], w[winsz1-i] );
    r[i] = max(r[i-1], w[winsz1+i]);
  }

  for (i = 0; i< winsz; i++){
        w[i+winsz2] = max(s[i], r[i]);
  }
}

// Perform 1D morphological dilation along X axis
// This kernel performs dilation along the horizontal (X) axis of images using
// the 1D dilation helper function. It processes images in chunks for efficiency.
// 
// Parameters:
//   im1:        Input image stack (float16 array)
//   out:        Output dilated image stack (float16 array)
//   winszX:     Window size for X-axis dilation
//   winszX2:    Half window size for X-axis dilation (winszX/2)
//   imszx:      Image width in pixels
//   imszy:      Image height in pixels
//   padx:       Padding size in x dimension
//   pady:       Padding size in y dimension
//   s:          Local memory scratch array for left-to-right pass
//   r:          Local memory scratch array for right-to-left pass
//   w:          Local memory working array
//
// The kernel processes each row of each image independently, applying 1D dilation
// along the horizontal direction. This is used as part of separable 2D dilation
// where X and Y directions are processed separately.
__kernel void morphDilateKernelX( __global const float16 *im1, __global float16 *out,
                        const unsigned long winszX, const unsigned long winszX2,
                        const unsigned long imszx, const unsigned long imszy,
                        const unsigned long padx, const unsigned long pady,
                        __local float16 *s, __local float16 *r, __local float16 *w)
{
  // IDs of work-item represent x and y coordinates in image
  const unsigned long int chunknum = get_global_id(0);
  const unsigned long int y = get_global_id(1)+pady;
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);
  
  unsigned long int i;
  unsigned long int indx;


  for (i=0; i<winszX; i++){
    s[i] = (float16) (-1.0e12f);
    r[i] = (float16) (-1.0e12f);
  }

  for (i =0; i<(winszX*2-1); ++i){
    indx = ((padx + chunknum*winszX - winszX2 +i)  + imszx*y)*nImChunk + z;
    w[i] = im1[indx];
  }

  morphDilateKernel(winszX, winszX2,s, r,w);
  for (i =winszX2; i<(winszX2+winszX); ++i){
    indx = ((padx + chunknum*winszX - winszX2 +i)  + imszx*y)*nImChunk + z;
    out[indx] = w[i];
  }

}


// Perform 1D morphological dilation along Y axis
// This kernel performs dilation along the vertical (Y) axis of images using
// the 1D dilation helper function. It processes images in chunks for efficiency.
// 
// Parameters:
//   im1:        Input image stack (float16 array)
//   out:        Output dilated image stack (float16 array)
//   winszY:     Window size for Y-axis dilation
//   winszY2:    Half window size for Y-axis dilation (winszY/2)
//   imszx:      Image width in pixels
//   imszy:      Image height in pixels
//   padx:       Padding size in x dimension
//   pady:       Padding size in y dimension
//   s:          Local memory scratch array for left-to-right pass
//   r:          Local memory scratch array for right-to-left pass
//   w:          Local memory working array
//
// The kernel processes each column of each image independently, applying 1D dilation
// along the vertical direction. This is used as part of separable 2D dilation
// where X and Y directions are processed separately for better performance than
// a full 2D kernel would provide.
__kernel void morphDilateKernelY( __global const float16 *im1, __global float16 *out,
                        const unsigned long winszY, const unsigned long winszY2,
                        const unsigned long imszx, const unsigned long imszy,
                        const unsigned long padx, const unsigned long pady,
                        __local float16 *s, __local float16 *r, __local float16 *w)
{
  // IDs of work-item represent x and y coordinates in image
  const unsigned long int x = get_global_id(0) + padx;
  const unsigned long int chunknum = get_global_id(1);
  const unsigned long int z = get_global_id(2);
  const unsigned long int nImChunk = get_global_size(2);

  //float s[128];
  //float r[128];
  //float w[256];


  unsigned long int i;
  unsigned long int indx; ;

  for (i=0; i<winszY; i++){
    s[i] = -1.0e12;
    r[i] = -1.0e12;
  }

  for (i =0; i<(winszY*2-1); ++i){
    indx = (x + (pady + chunknum*winszY - winszY2 +i) * imszx)*nImChunk + z;
    w[i] = im1[indx];
  }

  morphDilateKernel(winszY, winszY2,s, r,w);
  for (i =winszY2; i<(winszY2+winszY); ++i){
    indx = (x + (pady + chunknum*winszY - winszY2 +i) * imszx)*nImChunk + z;
    out[indx] = w[i];
  }

}


