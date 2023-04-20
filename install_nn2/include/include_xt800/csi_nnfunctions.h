/*
 * Copyright (C) 2016-2022 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Title:        csi_nnfunctions.h
 * Description:  Public header file for CSI NN Library
 *
 * -------------------------------------------------------------------- */

#ifndef INCLUDE_INCLUDE_XT800_CSI_NNFUNCTIONS_H_
#define INCLUDE_INCLUDE_XT800_CSI_NNFUNCTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "csi_instance.h"
#include "csi_nnsupportfunctions.h"

/**
 * @brief Struct for specifying activation function types
 *
 */
typedef enum {
    CSKY_SIGMOID = 0, /**< Sigmoid activation function */
    CSKY_TANH = 1,    /**< Tanh activation function */
} csi_nn_activation_type;

/**
 * @brief Basic Q7 convolution function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 */

void csi_convolve_HWC_q7_basic(const q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in,
                               const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel,
                               const uint16_t padding, const uint16_t stride, const q7_t *bias,
                               const uint16_t bias_shift, const uint16_t out_shift, q7_t *Im_out,
                               const uint16_t dim_im_out, q15_t *bufferA);

/**
 * @brief Basic Q15 convolution function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 */

void csi_convolve_HWC_q15_basic(const q15_t *Im_in, const uint16_t dim_im_in,
                                const uint16_t ch_im_in, const q15_t *wt, const uint16_t ch_im_out,
                                const uint16_t dim_kernel, const uint16_t padding,
                                const uint16_t stride, const q15_t *bias, const uint16_t bias_shift,
                                const uint16_t out_shift, q15_t *Im_out, const uint16_t dim_im_out,
                                q15_t *bufferA);

void csi_convolve_HWC_q15_fast(const q15_t *Im_in, const uint16_t dim_im_in,
                               const uint16_t ch_im_in, const q15_t *wt, const uint16_t ch_im_out,
                               const uint16_t dim_kernel, const uint16_t padding,
                               const uint16_t stride, const q15_t *bias, const uint16_t bias_shift,
                               const uint16_t out_shift, q15_t *Im_out, const uint16_t dim_im_out,
                               q15_t *bufferA);

/**
 * @brief Fast Q7 convolution function (non-sqaure shape)
 * @param[in]       Im_in        pointer to input tensor
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       dim_im_in_y  input tensor dimention y
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x filter kernel size x
 * @param[in]       dim_kernel_y filter kernel size y
 * @param[in]       padding_x    padding size x
 * @param[in]       padding_y    padding size y
 * @param[in]       stride_x     convolution stride x
 * @param[in]       stride_y     convolution stride y
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in]       dim_im_out_y output tensor dimension y
 * @param[in,out]   bufferA      pointer to buffer space for input
 * @return          none.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 */

void csi_convolve_HWC_q7_fast_nonsquare(
    const q7_t *Im_in, const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
    const uint16_t ch_im_in, const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel_x,
    const uint16_t dim_kernel_y, const uint16_t padding_x, const uint16_t padding_y,
    const uint16_t stride_x, const uint16_t stride_y, const q7_t *bias, const uint16_t bias_shift,
    const uint16_t out_shift, q7_t *Im_out, const uint16_t dim_im_out_x,
    const uint16_t dim_im_out_y, q15_t *bufferA);

/**
 * @brief Fast Q7 version of 1x1 convolution (non-sqaure shape)
 * @param[in]       Im_in        pointer to input tensor
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       dim_im_in_y  input tensor dimention y
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x filter kernel size x
 * @param[in]       dim_kernel_y filter kernel size y
 * @param[in]       padding_x    padding size x
 * @param[in]       padding_y    padding size y
 * @param[in]       stride_x     convolution stride x
 * @param[in]       stride_y     convolution stride y
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in]       dim_im_out_y output tensor dimension y
 * @param[in,out]   bufferA      pointer to buffer space for input
 * @return          none.
 *
 * This function implement convolution with 1x1 kernel size (i.e., dim_kernel_x=1
 * and dim_kernel_y=1). It can be used for
 * second half of MobileNets after depthwise separable convolution.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 */
void csi_convolve_1x1_HWC_q7_fast(const q7_t *Im_in, const uint16_t dim_im_in_x,
                                  const uint16_t dim_im_in_y, const uint16_t ch_im_in,
                                  const q7_t *wt, const uint16_t ch_im_out, const q7_t *bias,
                                  const uint16_t bias_shift, const uint16_t out_shift, q7_t *Im_out,
                                  const uint16_t dim_im_out_x, const uint16_t dim_im_out_y,
                                  q15_t *bufferA);

/**
 * @brief Q7 version of convolution for RGB image
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 * This kernel is written exclusively for convolution with ch_im_in
 * equals 3. This applies on the first layer of CNNs which has input
 * image with RGB format.
 */

void csi_convolve_HWC_q7_RGB(const q7_t *Im_in, const uint16_t dim_im_in, const q7_t *wt,
                             const uint16_t ch_im_out, const uint16_t dim_kernel,
                             const uint16_t padding, const uint16_t stride, const q7_t *bias,
                             const uint16_t bias_shift, const uint16_t out_shift, q7_t *Im_out,
                             const uint16_t dim_im_out, q15_t *bufferA);

/**
 * @brief Q7 depthwise separable convolution function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @return          none.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 2
 *   ch_im_out is multiple of 2
 */

void csi_depthwise_separable_conv_HWC_q7(const q7_t *Im_in, const uint16_t dim_im_in,
                                         const uint16_t ch_im_in, const q7_t *wt,
                                         const uint16_t ch_im_out, const uint16_t dim_kernel,
                                         const uint16_t padding, const uint16_t stride,
                                         const q7_t *bias, const uint16_t bias_shift,
                                         const uint16_t out_shift, q7_t *Im_out,
                                         const uint16_t dim_im_out, q15_t *bufferA);

/**
 * @brief Q7 depthwise separable convolution function (non-square shape)
 * @param[in]       Im_in         pointer to input tensor
 * @param[in]       dim_im_in_x   input tensor dimention x
 * @param[in]       dim_im_in_y   input tensor dimention y
 * @param[in]       ch_im_in      number of input tensor channels
 * @param[in]       wt            pointer to kernel weights
 * @param[in]       ch_im_out     number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x  filter kernel size x
 * @param[in]       dim_kernel_y  filter kernel size y
 * @param[in]       padding_x     padding sizes x
 * @param[in]       padding_y     padding sizes y
 * @param[in]       stride_x      convolution stride x
 * @param[in]       stride_y      convolution stride y
 * @param[in]       bias          pointer to bias
 * @param[in]       bias_shift    amount of left-shift for bias
 * @param[in]       out_shift     amount of right-shift for output
 * @param[in,out]   Im_out        pointer to output tensor
 * @param[in]       dim_im_out_x  output tensor dimension x
 * @param[in]       dim_im_out_y  output tensor dimension y
 * @param[in,out]   bufferA       pointer to buffer space for input
 * @return          none.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 2
 *   ch_im_out is multiple of 2
 */
void csi_depthwise_separable_conv_HWC_q7_nonsquare(
    const q7_t *Im_in, const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
    const uint16_t ch_im_in, const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel_x,
    const uint16_t dim_kernel_y, const uint16_t padding_x, const uint16_t padding_y,
    const uint16_t stride_x, const uint16_t stride_y, const q7_t *bias, const uint16_t bias_shift,
    const uint16_t out_shift, q7_t *Im_out, const uint16_t dim_im_out_x,
    const uint16_t dim_im_out_y, q15_t *bufferA);

/**
 * @brief Q7 basic fully-connected layer function
 * @param[in]       pV          pointer to input vector
 * @param[in]       pM          pointer to matrix weights
 * @param[in]       dim_vec     length of the vector
 * @param[in]       num_of_rows number of rows in weight matrix
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in]       bias        pointer to bias
 * @param[in,out]   pOut        pointer to output vector
 * @return          none.
 */

void csi_fully_connected_q7(const q7_t *pV, const q7_t *pM, const uint16_t dim_vec,
                            const uint16_t num_of_rows, const uint16_t bias_shift,
                            const uint16_t out_shift, const q7_t *bias, q7_t *pOut);

/**
 * @brief Q15 basic fully-connected layer function
 * @param[in]       pV          pointer to input vector
 * @param[in]       pM          pointer to matrix weights
 * @param[in]       dim_vec     length of the vector
 * @param[in]       num_of_rows number of rows in weight matrix
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in]       bias        pointer to bias
 * @param[in,out]   pOut        pointer to output vector
 * @return          none.
 *
 */

void csi_fully_connected_q15(const q15_t *pV, const q15_t *pM, const uint16_t dim_vec,
                             const uint16_t num_of_rows, const uint16_t bias_shift,
                             const uint16_t out_shift, const q15_t *bias, q15_t *pOut);

/**
 * @brief Mixed Q15-Q7 fully-connected layer function
 * @param[in]       pV          pointer to input vector
 * @param[in]       pM          pointer to matrix weights
 * @param[in]       dim_vec     length of the vector
 * @param[in]       num_of_rows number of rows in weight matrix
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in]       bias        pointer to bias
 * @param[in,out]   pOut        pointer to output vector
 * @return          none.
 *
 */

void csi_fully_connected_mat_q7_vec_q15(const q15_t *pV, const q7_t *pM, const uint16_t dim_vec,
                                        const uint16_t num_of_rows, const uint16_t bias_shift,
                                        const uint16_t out_shift, const q7_t *bias, q15_t *pOut);

/**
 * @brief Q7 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @return none.
 */

void csi_relu_q7(q7_t *data, uint16_t size);

/**
 * @brief Q15 RELU function
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @return none.
 */

void csi_relu_q15(q15_t *data, uint16_t size);

/**
 * @brief Q7 neural network activation function using direct table look-up
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
 * @param[in]       type        type of activation functions
 * @return none.
 */

void csi_nn_activations_direct_q7(q7_t *data, uint16_t size, uint16_t int_width,
                                  csi_nn_activation_type type);

/**
 * @brief Q15 neural network activation function using direct table look-up
 * @param[in,out]   data        pointer to input
 * @param[in]       size        number of elements
 * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
 * @param[in]       type        type of activation functions
 * @return none.
 */

void csi_nn_activations_direct_q15(q15_t *data, uint16_t size, uint16_t int_width,
                                   csi_nn_activation_type type);

/**
 * @brief Q7 max pooling function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @param[in,out]   Im_out      pointer to output tensor
 * @return none.
 *
 */

void csi_maxpool2d_q7_HWC(q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in,
                          const uint16_t dim_kernel, const uint16_t padding, const uint16_t stride,
                          const uint16_t dim_im_out, q7_t *bufferA, q7_t *Im_out);

/**
 * @brief Q7 average pooling function
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in   input tensor dimention
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       dim_kernel  filter kernel size
 * @param[in]       padding     padding sizes
 * @param[in]       stride      convolution stride
 * @param[in]       dim_im_out  output tensor dimension
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @param[in,out]   Im_out      pointer to output tensor
 * @return none.
 *
 */

void csi_avepool_q7_HWC(q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in,
                        const uint16_t dim_kernel, const uint16_t padding, const uint16_t stride,
                        const uint16_t dim_im_out, q7_t *bufferA, q7_t *Im_out);

void csi_avepool_q7_HWC_nonsquare(q7_t *Im_in,                  // input image
                                  const uint16_t dim_im_in_x,   // input image dimension
                                  const uint16_t dim_im_in_y,   // input image dimension
                                  const uint16_t ch_im_in,      // number of input image channels
                                  const uint16_t dim_kernel_x,  // window kernel size
                                  const uint16_t dim_kernel_y,  // window kernel size
                                  const uint16_t padding_x,     // padding sizes
                                  const uint16_t padding_y,     // padding sizes
                                  const uint16_t stride_x,      // stride
                                  const uint16_t stride_y,      // stride
                                  const uint16_t dim_im_out_x,  // output image dimension
                                  const uint16_t dim_im_out_y,  // output image dimension
                                  q7_t *bufferA,                // a buffer for local storage
                                  q7_t *Im_out,                 // output feature
                                  const uint16_t out_lshift);   // output left shift (scaling)

/**
 * @brief Q7 softmax function
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimention
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 */

void csi_softmax_q7(const q7_t *vec_in, const uint16_t dim_vec, q7_t *p_out);

/**
 * @brief Q15 softmax function
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimention
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 */

void csi_softmax_q15(const q15_t *vec_in, const uint16_t dim_vec, q15_t *p_out);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_INCLUDE_XT800_CSI_NNFUNCTIONS_H_
