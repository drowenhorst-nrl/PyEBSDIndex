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
The US Naval Research Laboratory Date: 22 May 2024

For more info see
Patrick T. Brewick, Stuart I. Wright, David J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.
"""Non-local pattern averaging and re-indexing (NLPAR)."""
*/


/**
 * Sums all 16 elements of a float16 vector.
 * 
 * This utility function iterates through all 16 components (s0-sf) of a float16 vector
 * and returns their sum. This is used throughout the OpenCL kernels to aggregate
 * similarity scores across pattern chunks.
 * 
 * @param v1 Pointer to a float16 vector containing 16 floating-point values
 * @return Sum of all 16 elements as a float
 */
float sum16(const float16 *v1);
float sum16(const float16 *v1){
     float sum = 0.0;
     sum += v1[0].s0;
     sum += v1[0].s1;
     sum += v1[0].s2;
     sum += v1[0].s3;
     sum += v1[0].s4;
     sum += v1[0].s5;
     sum += v1[0].s6;
     sum += v1[0].s7;
     sum += v1[0].s8;
     sum += v1[0].s9;
     sum += v1[0].sa;
     sum += v1[0].sb;
     sum += v1[0].sc;
     sum += v1[0].sd;
     sum += v1[0].se;
     sum += v1[0].sf;
     return sum;

}


/**
 * Prints all 16 elements of a float16 vector for debugging purposes.
 * 
 * This utility function outputs all 16 components (s0-sf) of a float16 vector
 * to stdout, separated by commas. Used for debugging and verifying data values.
 */
void print16(const float16 v1);
void print16(const float16 v1){
     
     printf( "%f, ", v1.s0);
     printf( "%f, ", v1.s1);
     printf( "%f, ", v1.s2);
     printf( "%f, ", v1.s3);
     printf( "%f, ", v1.s4);
     printf( "%f, ", v1.s5);
     printf( "%f, ", v1.s6);
     printf( "%f, ", v1.s7);
     printf( "%f, ", v1.s8);
     printf( "%f, ", v1.s9);
     printf( "%f, ", v1.sa);
     printf( "%f, ", v1.sb);
     printf( "%f, ", v1.sc);
     printf( "%f, ", v1.sd);
     printf( "%f, ", v1.se);
     printf( "%f", v1.sf);

}


/**
 * OpenCL kernel to convert 8-bit unsigned char data to float.
 * 
 * This kernel takes an array of 8-bit unsigned integers and converts each element
 * to a floating-point value. Used for loading EBSD pattern data from 8-bit images.
 * 
 * @param datain Input array of unsigned char (8-bit) values
 * @param dataout Output array where each element is converted to float
 */
__kernel void nlloadpat8bit(__global uchar *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  uchar imVal =  datain[x];
  float imValflt = convert_float(imVal);
  dataout[x] = imValflt;
}

/**
 * OpenCL kernel to convert 16-bit unsigned short data to float.
 * 
 * This kernel takes an array of 16-bit unsigned integers and converts each element
 * to a floating-point value. Used for loading EBSD pattern data from 16-bit images.
 * 
 * @param datain Input array of unsigned short (16-bit) values
 * @param dataout Output array where each element is converted to float
 */
__kernel void nlloadpat16bit(__global ushort *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  ushort imVal = datain[x];
  float imValflt = convert_float(imVal);
  dataout[x] = imValflt;
}

/**
 * OpenCL kernel to pass through 32-bit float data.
 * 
 * This simple kernel copies float data from input to output without modification.
 * Used when the input data is already in 32-bit float format.
 * 
 * @param datain Input array of float (32-bit) values
 * @param dataout Output array receiving the same float values
 */
__kernel void nlloadpat32flt(__global float *datain, __global float *dataout){
  const unsigned long int x = get_global_id(0);
  dataout[x] = datain[x];
}

/**
 * OpenCL kernel to calculate sigma values for pattern similarity metrics.
 * 
 * This kernel computes the local variance (sigma) of pattern intensities in a neighborhood
 * around each pixel. It calculates how much patterns vary from their neighbors, which is used
 * to determine appropriate weighting in the non-local means averaging.
 * 
 * The kernel:
 * 1. Computes squared differences between the center pattern and neighboring patterns
 * 2. Aggregates these differences across all pattern chunks (ndatchunk)
 * 3. Finds the minimum variance for each pixel
 * 4. Stores results in output arrays (sig, dg, ng)
 * 
 * @param data Input array of EBSD pattern data (float values)
 * @param mask Input mask array indicating valid/invalid regions
 * @param sig Output array for sigma values (local variance)
 * @param dg Output array for difference scores between patterns
 * @param ng Output array for neighbor counts used in calculations
 * @param nn Search radius/neighborhood size parameter (half-width of search window)
 * @param ndatchunk Number of pattern chunks being processed
 * @param npatpoint Number of data points per pattern chunk
 * @param maxlim Maximum intensity threshold for masking out saturated pixels
 */
__kernel void calcsigma( __global float *data, __global float16 *mask, 
      __global float *sig, 
      __global float *dg, __global float *ng,
      const long  nn, const long ndatchunk, const long npatpoint, const float maxlim){
  // IDs of work-item represent x and y coordinates in image
  const long x = get_global_id(0);
  const long y = get_global_id(1);
  const long ncol = get_global_size(0);
  const long nrow = get_global_size(1);
  const long indx_xy = x+y*ncol;
  
  const long indx0 = npatpoint * indx_xy;

  long i, j, z;
  long indx_j, indx_ij, count; 

  long nnn = (2*nn+1) * (2*nn+1);

  float16 d1, d0; 
  float16 maskchunk; 
  float16 mask0, mask1;
  float dd; 


  float d[256];
  float n[256];
 
  
  for(j=0; j < nnn; ++j){
      d[j] = 0.0;
      n[j] = 1.0e-6; 
  }

  ;

  for(z = 0; z<ndatchunk; ++z){
      count = 0; 
      
      
      d0 =  *(__global float16*) (data + indx0+16*z); 
      //if (z == 0){print16(d0);}
      //
      maskchunk = mask[z];
      //print16(maskchunk);
      mask0 =  d0 < (float16) (maxlim) ? (float16) (1.0): (float16) (0.0);
      mask0 =  maskchunk > (float16) 1.e-3 ? mask0 : (float16) 0.0; 
      //mask0 = select((float16) (1.0), (float16) (0.0), isless( d0, (float16) maxlim));
      
      for(j=y-nn; j<=y+nn; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-nn; i<=x+nn; ++i){      
            
              indx_ij =  (i >= 0) ? (i): abs(i);
              indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
              indx_ij = npatpoint*(indx_ij + indx_j);
              
              mask1 = mask0;
              d1 =  *(__global float16*) (data + indx_ij+16*z); 
              

              mask1 =  d1 < (float16) (maxlim) ? mask1 : (float16) (0.0);
              //mask1 = select((float16) (1.0), (float16) (0.0), isgreater(mask0, (float16)(1e-6)) && isgreater(mask1,(float16)(1e-6)));
              //printf("%*s\n", 'here');
              
              d1 = (d0-d1);
              d1 *= d1 ; 
              d1 *= mask1; 

              dd = sum16(&d1);
              dd = (indx_ij == indx0) ? -1*dd : dd; // mark the center point 
              
              d[count] += dd;
              n[count] += sum16(&mask1);
              //d[count+nnn*indx_xy] += dd;
              //n[count+nnn*indx_xy] += sum16(&mask1);

              count += 1; 
          }
      }
      //printf("%d %f\n", count, n[count]);
  }


  float mind = 1e24; 
  float s0;  
  ; //nnn*(x+ncol*y);
  for(j=0; j<nnn; ++j){
    if (d[j] > 1e-5){ //sometimes EDAX collects the same pattern twice & catch the pixel of interest.
      s0 = d[j]/(2.0*n[j]); 
      if (s0<mind){
        mind = s0; 
      }
    } 
  }

  sig[indx_xy] = sqrt(mind);
  
  for(j=0; j<nnn; ++j){
    dg[j+nnn*indx_xy] = d[j];
    ng[j+nnn*indx_xy] = n[j];
  }

}


/**
 * OpenCL kernel to normalize difference arrays used in weighting calculations.
 * 
 * This kernel normalizes the accumulated differences (d) by incorporating:
 * - Local sigma values for each pixel
 * - Neighbor counts to account for varying numbers of valid neighbors
 * 
 * The normalization formula incorporates the non-local means weighting:
 *   normalized_diff = (diff - n*sigma²) / (σ√(2n))
 * 
 * This weighting scheme gives higher importance to patterns that are:
 * - More similar (lower diff)
 * - From regions with more neighbors
 * - In areas of lower variance (more consistent patterns)
 * 
 * @param sigma Input array of sigma values (local variance) for each pixel
 * @param n Input array of neighbor counts used in calculations
 * @param d Output/working array for normalized differences (also used as input)
 * @param nn Search radius parameter defining the neighborhood size
 */
__kernel void normd(
    const __global float *sigma, 
    const __global float *n, 
     __global float *d,
     const long nn){

    const long x = get_global_id(0);
    const long y = get_global_id(1);
    const long ncol = get_global_size(0);
    const long nrow = get_global_size(1);
    const long indx_xy = x+y*ncol;

    long i, j, q;
    long indx_j, indx_ij; 

    long nnn = (2*nn+1) * (2*nn+1);
    //printf("%d\n", nnn) ;
    float sigma_xy = sigma[indx_xy];  
    sigma_xy *= sigma_xy;
    // printf("%f", sigma_xy);
    float sigma_ij, n_ij, d_ij;  
    q = 0; 
    for(j=y-nn; j<=y+nn; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-nn; i<=x+nn; ++i){  
              
               indx_ij =  (i >= 0) ? (i): abs(i);
               indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
               indx_ij =  (indx_ij + indx_j);
               sigma_ij = sigma[indx_ij];
               sigma_ij *= sigma_ij;
              
               sigma_ij = sigma_ij + sigma_xy;
               
               d_ij = d[q+nnn*indx_xy];
               n_ij = n[q+nnn*indx_xy];    
               if (n_ij > 1.0e-3){           
                  d_ij -= n_ij*sigma_ij;
                  d_ij *= 1.0/( sigma_ij * sqrt(2.0*n_ij) ); 
                  
                  d[q+nnn*indx_xy] = d_ij;    
                }
               q += 1.0; 
           }
      } 
      //printf("%d, %d, %d\n",nn, q, nnn);     
}




/**
 * Main OpenCL kernel implementing the Non-local Pattern Averaging and Re-indexing (NLPAR) algorithm.
 * 
 * This is the primary kernel that performs pattern averaging by:
 * 1. Computing similarity weights between a center pixel and its neighbors
 * 2. Applying exponential weighting based on pattern differences
 * 3. Averaging patterns using these weights to improve signal-to-noise ratio
 * 
 * The NLPAR algorithm works by:
 * - For each pixel at (x,y), examining a search window of size (2*sr+1)×(2*sr+1)
 * - Computing similarity scores between the center pattern and neighboring patterns
 * - Calculating weights using exponential decay: w = exp(-λ·diff)
 * - Averaging patterns weighted by these similarity scores
 * 
 * The kernel processes pattern data in chunks (ndatchunk) and outputs averaged patterns.
 * 
 * @param data Input array of EBSD pattern data (float values)
 * @param mask Input mask array indicating valid/invalid regions for each pattern chunk
 * @param sigma Input array of pre-computed sigma values (local variance) for weighting
 * @param crlimits Array containing calculation limits: [x_offset, y_offset, ncols, nrows]
 * @param dataout Output array receiving the averaged pattern data
 * @param sr Search radius parameter defining neighborhood size (half-width of search window)
 * @param ndatchunk Number of pattern chunks being processed
 * @param npatpoint Number of data points per pattern chunk
 * @param maxlim Maximum intensity threshold for masking out saturated pixels
 * @param lam2 Lambda parameter controlling the exponential weighting strength (weight decay rate)
 * @param dthresh Threshold for difference values below which weights are set to zero
 * @param diff_offset Small offset added when center pixel is detected (to avoid self-weighting)
 */
__kernel void calcnlpar( 
      const __global float *data, 
      const __global float16 *mask, 
      const __global float *sigma, 
      const __global long *crlimits, 
      __global float *dataout, 
      const long sr, 
      const long ndatchunk, 
      const long npatpoint, 
      const float maxlim, 
      const float lam2, 
      const float dthresh,
      const float diff_offset){
  //IDs of work-item represent x and y coordinates in image
  //const long4 calclim =  crlimits[0];
  const long x = get_global_id(0)+crlimits[0];
  const long y = get_global_id(1)+crlimits[1];
  const long ncol = crlimits[2]; //get_global_size(0);
  const long nrow = crlimits[3];//get_global_size(1);
  const long indx_xy = x+y*ncol;
  //printf("%d\n", indx_xy);
  const long indx0 = npatpoint * indx_xy;
  //printf("%d, %d, %d, %d\n", x,y,ncol, nrow); 
  long i, j, z;
  long indx_j, indx_ij, count; 

  long nnn = (2*sr+1) * (2*sr+1);

  float16 d1, d0; 
  float16 mask0, mask1;
  float dd, nn, sigma_ij, norm; 
  float sigma0 = sigma[indx_xy];
  sigma0 *= sigma0; 
  //float16* writeloc; 

  float d[512]; // taking a risk here that noone will want a SR > 10
  float n[512]; 
  float diff_step[512];

  
  for(j=0; j < nnn; ++j){
      d[j] = 0.0;
      n[j] = 1.0e-6; 
      diff_step[j] = 0.0 ;
  }


  for(z = 0; z<ndatchunk; ++z){
      count = 0; 
      
      
      d0 =  *(__global float16*) (data + indx0+16*z); 
      
      mask0 = mask[z];
      mask0 =  (mask0 > (float16) 1.e-3) ? (float16) 1.0 : (float16) 0.0; 
      mask0 =  d0 < (float16) (maxlim) ? mask0 : (float16) (0.0);
      
      
      for(j=y-sr; j<=y+sr; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-sr; i<=x+sr; ++i){
            
              indx_ij =  (i >= 0) ? (i): abs(i);
              indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
              
              indx_ij = (indx_ij + indx_j);
              
              indx_ij *= npatpoint;
              
              mask1 = mask0;
              d1 =  *(__global float16*) (data + indx_ij+16*z); 
              

              mask1 =  d1 < (float16) (maxlim) ? mask1 : (float16) (0.0);
              //mask1 = select((float16) (1.0), (float16) (0.0), isgreater(mask0, (float16)(1e-6)) && isgreater(mask1,(float16)(1e-6)));
              //printf("%*s\n", 'here');
              
              d1 = (d0-d1);
              d1 *= d1 ; 
              d1 *= mask1; 

              dd = sum16(&d1);
              if (indx_ij == indx0) {
                dd = -1.0;
                diff_step[count] +=  diff_offset;
              } else{
                dd = dd;
                diff_step[count] = 0.0; //diff_offset;
              }
              //dd = (indx_ij == indx0) ? -1.0 : dd; // mark the center point 
              
              d[count] += dd;
              n[count] += sum16(&mask1);
              //d[count+nnn*indx_xy] += dd;
              //n[count+nnn*indx_xy] += sum16(&mask1);

              count += 1; 
          }
      }
      
  }
// calculate the weights
  count = 0; 
  float sum = 0.0; 
  nn = 1.0;
  for(j=y-sr; j<=y+sr; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-sr; i<=x+sr; ++i){
              
              dd = d[count]; 
              if (dd > 1.e-3){
                indx_ij =  (i >= 0) ? (i): abs(i);
                indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
                
                indx_ij = (indx_ij + indx_j);
                
                sigma_ij = sigma[indx_xy];
                sigma_ij *= sigma_ij; 
                nn = n[count]; 
                dd -= nn*(sigma_ij+sigma0);
                
                norm = (sigma0 + sigma_ij)*sqrt(2.0*nn);
                if (norm > 1.0e-8){
                  dd /= norm;  
                } else {
                  dd = 1e6*nn;
                } 

              } else {
                dd = 0.0;
                nn = 1.0;
              }
              
              dd -= dthresh;

              dd = dd >= 0.0 ? dd : 0.0; 

              dd = exp(-1.0*dd*lam2); 
              dd += diff_step[count]; 
              sum += dd; 
              d[count] = dd;

              count += 1; 
              
          }
  }
  
  for (j =0;j<nnn;j++){d[j] *= 1.0/sum; }

// now one more time through the loops to average across the patterns with the weights  
// and place in the output array
  for(z = 0; z<ndatchunk; ++z){
      count = 0; 
      d0 =  *(__global float16*) (data + indx0+16*z); 
      
      mask0 = mask[z];
      mask0 =  (mask0 > (float16) -1.e-3) ? (float16) 1.0 : (float16) 0.0; 
      
      for(j=y-sr; j<=y+sr; ++j){
          
          indx_j =  (j >= 0) ? (j): abs(j);
          indx_j =  (indx_j < nrow) ? (indx_j): nrow - (indx_j -nrow +1);
          indx_j = ncol * indx_j;
          
          for(i=x-sr; i<=x+sr; ++i){
            
              indx_ij =  (i >= 0) ? (i): abs(i);
              indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
              
              indx_ij = (indx_ij + indx_j);
              
              indx_ij *= npatpoint;
              
              
              d1 =  *(__global float16*) (data + indx_ij+16*z); 
              
              
              d1 *= (float16) d[count]; 
              d1 *= mask0; 
              
              *(__global float16*) (dataout + indx0+16*z) += d1; 
              
              //writeadd16(&(dataout[indx0+16*z]), &d1 );

              //d[count+nnn*indx_xy] += dd;
              //n[count+nnn*indx_xy] += sum16(&mask1);

              count += 1; 
          }
      }
  }



}





