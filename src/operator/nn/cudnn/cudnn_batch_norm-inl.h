/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_batch_norm-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_INL_H_
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "../batch_norm-inl.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 4
namespace cudnnbatchnorm {
enum CuDNNBatchNormOpInputs {kData, kGamma, kBeta};
enum CuDNNBatchNormOpOutputs {kOut, kMean, kInvVar};
enum CuDNNBatchNormOpAuxiliary {kMovingMean, kMovingInvVar};
}  // namespace cudnnbatchnorm

#if defined(__CUDACC__)
template<typename DType>
class CuDNNBatchNormOp {
 public:
  CuDNNBatchNormOp() {
    using namespace mshadow;
    dtype_ = DataType<DType>::kCudnnFlag;
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? kFloat32 : DataType<DType>::kFlag;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&io_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&mean_desc_));
  }

  void Init(const BatchNormParam &param) {
    CHECK_GE(param.eps, CUDNN_BN_MIN_EPSILON)
     << "CuDNN requires eps to be no less than " << CUDNN_BN_MIN_EPSILON;
    this->param_ = param;
  }

  ~CuDNNBatchNormOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(io_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(mean_desc_));
  }

  void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(aux_states.size(), 2U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
    }
    CHECK_EQ(req[cudnnbatchnorm::kOut], kWriteTo);
    CHECK_GE(in_data[cudnnbatchnorm::kData].ndim(), 2);
    CHECK_LE(in_data[cudnnbatchnorm::kData].ndim(), 4);

    Init(in_data[cudnnbatchnorm::kData]);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> x =
      in_data[cudnnbatchnorm::kData].get_with_shape<gpu, 4, DType>(shape_, s);

    Tensor<gpu, 4, DType> y =
      out_data[cudnnbatchnorm::kOut].get_with_shape<gpu, 4, DType>(shape_, s);
#if CUDNN_VERSION >= 7002
    auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    auto mode = CUDNN_BATCHNORM_SPATIAL;
#endif

    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> beta =
        in_data[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> moving_mean =
        aux_states[cudnnbatchnorm::kMovingMean]
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> moving_inv_var =
        aux_states[cudnnbatchnorm::kMovingInvVar]
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      typename DataType<DType>::ScaleType a = 1.0f;
      typename DataType<DType>::ScaleType b = 0.0f;

      if (param_.fix_gamma) gamma = 1.f;

      if (ctx.is_train) {
        Tensor<gpu, 1, DTypeParam> save_mean =
          out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
        Tensor<gpu, 1, DTypeParam> save_inv_var =
          out_data[cudnnbatchnorm::kInvVar]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
        CUDNN_CALL(cudnnBatchNormalizationForwardTraining(s->dnn_handle_,
                                                          mode,
                                                          &a,
                                                          &b,
                                                          io_desc_,
                                                          x.dptr_,
                                                          io_desc_,
                                                          y.dptr_,
                                                          mean_desc_,
                                                          gamma.dptr_,
                                                          beta.dptr_,
                                                          1 - param_.momentum,
                                                          moving_mean.dptr_,
                                                          moving_inv_var.dptr_,
                                                          param_.eps,
                                                          save_mean.dptr_,
                                                          save_inv_var.dptr_));
      } else {
        CUDNN_CALL(cudnnBatchNormalizationForwardInference(s->dnn_handle_,
                                                           CUDNN_BATCHNORM_SPATIAL,
                                                           &a,
                                                           &b,
                                                           io_desc_,
                                                           x.dptr_,
                                                           io_desc_,
                                                           y.dptr_,
                                                           mean_desc_,
                                                           gamma.dptr_,
                                                           beta.dptr_,
                                                           moving_mean.dptr_,
                                                           moving_inv_var.dptr_,
                                                           param_.eps));
      }
    })
  }

  void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);
    CHECK(ctx.is_train && !param_.use_global_stats)
        << "use global statistics is not yet supported in CuDNNBatchNorm";

    Init(in_data[cudnnbatchnorm::kData]);
    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 4, DType> x =
      in_data[cudnnbatchnorm::kData].get_with_shape<gpu, 4, DType>(shape_, s);
    Tensor<gpu, 4, DType> dx =
      in_grad[cudnnbatchnorm::kData].get_with_shape<gpu, 4, DType>(shape_, s);
    Tensor<gpu, 4, DType> dy =
      out_grad[cudnnbatchnorm::kOut].get_with_shape<gpu, 4, DType>(shape_, s);

#if CUDNN_VERSION >= 4007
#if CUDNN_VERSION >= 7002
    auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    auto mode = CUDNN_BATCHNORM_SPATIAL;
#endif
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> dbeta =
        in_grad[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> dgamma =
        in_grad[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> save_mean =
        out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> save_inv_var =
        out_data[cudnnbatchnorm::kInvVar].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);

      typename DataType<DType>::ScaleType a = 1.0f;
      typename DataType<DType>::ScaleType b = 0.0f;
      typename DataType<DType>::ScaleType b_add = 1.0f;
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);

      if (param_.fix_gamma) gamma = 1.f;

      CUDNN_CALL(cudnnBatchNormalizationBackward(
        s->dnn_handle_,
        mode,
        &a,
        &b,
        &a,
        req[cudnnbatchnorm::kGamma] == kWriteTo ? &b: &b_add,
        io_desc_,
        x.dptr_,
        io_desc_,
        dy.dptr_,
        io_desc_,
        dx.dptr_,
        mean_desc_,
        gamma.dptr_,
        dgamma.dptr_,
        dbeta.dptr_,
        param_.eps,
        save_mean.dptr_,
        save_inv_var.dptr_));
      if (param_.fix_gamma) dgamma = 0.f;
    })
#else  // CUDNN_VERSION < 4007
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> dbeta =
        in_grad[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> dgamma =
        in_grad[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> save_mean =
        out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);
      Tensor<gpu, 1, DTypeParam> save_inv_var =
        out_data[cudnnbatchnorm::kInvVar].get_with_shape<gpu, 1, DTypeParam>(Shape1(shape_[1]), s);

      typename DataType<DType>::ScaleType a = 1.0f;
      typename DataType<DType>::ScaleType b = 0.0f;
      typename DataType<DType>::ScaleType b_add = 1.0f;
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);

      if (param_.fix_gamma) gamma = 1.f;
      CUDNN_CALL(cudnnBatchNormalizationBackward(s->dnn_handle_,
                                                 CUDNN_BATCHNORM_SPATIAL,
                                                 &a,
                                                 &b,
                                                 io_desc_,
                                                 x.dptr_,
                                                 io_desc_,
                                                 dy.dptr_,
                                                 io_desc_,
                                                 dx.dptr_,
                                                 mean_desc_,
                                                 gamma.dptr_,
                                                 dgamma.dptr_,
                                                 dbeta.dptr_,
                                                 param_.eps,
                                                 save_mean.dptr_,
                                                 save_inv_var.dptr_));
      if (param_.fix_gamma) dgamma = 0.f;
    })
#endif
  }

 private:
  void Init(const TBlob &in_data) {
    for (int i = 0; i < 4; ++i) {
      if (i < in_data.ndim()) {
        shape_[i] = in_data.shape_[i];
      } else {
        shape_[i] = 1;
      }
    }

    CUDNN_CALL(cudnnSetTensor4dDescriptor(io_desc_,
                                          CUDNN_TENSOR_NCHW,
                                          dtype_,
                                          shape_[0],
                                          shape_[1],
                                          shape_[2],
                                          shape_[3]));
    CUDNN_CALL(cudnnDeriveBNTensorDescriptor(mean_desc_,
                                             io_desc_,
                                             CUDNN_BATCHNORM_SPATIAL));
  }

  cudnnDataType_t dtype_;
  int dtype_param_;
  cudnnTensorDescriptor_t io_desc_, mean_desc_;
  mshadow::Shape<4> shape_;
  BatchNormParam param_;
};
#endif  // defined(__CUDACC__)

#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_INL_H_