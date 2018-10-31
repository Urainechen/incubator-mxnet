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
 * \file quantized_concat.cc
 * \brief
 * \author Junming Chen
*/


#include "../mkldnn/mkldnn_ops-inl.h"
#include "../mkldnn/mkldnn_base-inl.h"
#include "../../../common/utils.h"
#include <mxnet/op_attr_types.h>
#include "../nn/concat-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "../nn/mkldnn/mkldnn_concat-inl.h"
#endif



namespace mxnet {
namespace op {

bool QuantizedConcatShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape) {
                            
  using namespace mshadow;
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args*3));
  TShape dshape;
  index_t size = 0;
  bool has_zero = false;
  int axis = -1;
  for (int i = 0; i < param_.num_args; ++i) {
    TShape tmp = (*in_shape)[i];
    if (tmp.ndim()) {
      axis = CheckAxis(param_.dim, tmp.ndim());
      has_zero = tmp[axis] == 0 || has_zero;
      size += tmp[axis];
      tmp[axis] = 0;
      shape_assign(&dshape, tmp);
    }
  }

  TShape tmp = (*out_shape)[0];
  if (tmp.ndim()) {
    axis = CheckAxis(param_.dim, tmp.ndim());
    tmp[axis] = 0;
    shape_assign(&dshape, tmp);
  }

  if (dshape.ndim() == 0) return false;

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (!has_zero) dshape[axis] = size;
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];

  return dshape.Size() != 0;
}


// Concat for RNN param deals with the reverse shape inference from output
// for the special case of concatenating RNN parameters.
// The first (and sometimes the second) input may be unknown on the target axis.
// If the two inputs are unknown, they always have the same shape.
// static bool  QuantizedRNNParamConcatShape(const nnvm::NodeAttrs& attrs,
//                                 std::vector<TShape> *in_shape,
//                                 std::vector<TShape> *out_shape) {
//   using namespace mshadow;
//   const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
//   CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args*3));
//   TShape dshape;
//   index_t size = 0;
//   int num_zero = 0;
//   int axis = -1;
//   for (int i = 0; i < param_.num_args; ++i) {
//     TShape tmp = (*in_shape)[i];
//     if (tmp.ndim()) {
//       axis = CheckAxis(param_.dim, tmp.ndim());
//       num_zero += tmp[axis] == 0;
//       size += tmp[axis];
//       tmp[axis] = 0;
//       shape_assign(&dshape, tmp);
//     }
//   }

//   TShape tmp = (*out_shape)[0];
//   if (tmp.ndim()) {
//     axis = CheckAxis(param_.dim, tmp.ndim());
//     tmp[axis] = 0;
//     shape_assign(&dshape, tmp);
//   }

//   if (dshape.ndim() == 0) return false;

//   for (int i = 0; i < param_.num_args; ++i) {
//     CHECK(shape_assign(&(*in_shape)[i], dshape))
//         << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
//   }

//   if (!num_zero) dshape[axis] = size;
//   CHECK(shape_assign(&(*out_shape)[0], dshape))
//       << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];
//   if ((*out_shape)[0][axis] != 0 && num_zero) {
//     int residual = (*out_shape)[0][axis] - size;
//     CHECK_GE(residual, 0)
//         << "Input size already exceeds output size. Residual: " << residual;
//     CHECK(num_zero <= 2 && num_zero >= 0)
//         << "Expecting 1 or 2 inputs that need shape inference. Got: " << num_zero;
//     bool need_infer = !(*out_shape)[0].Size();
//     for (int i = 0; i < num_zero; i++) {
//       (*in_shape)[i*2][axis] = residual / num_zero;
//       need_infer = need_infer || !(*in_shape)[i].Size();
//     }
//     return !need_infer;
//   }

//   return dshape.Size() != 0;
// }

static bool QuantizedConcatType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_type,
                       std::vector<int> *out_type) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_type->size(),static_cast<size_t>(param_.num_args*3));
  CHECK_EQ(out_type->size(), 3U);


#if MXNET_USE_MKLDNN ==1
 
    for(size_t i=param_.num_args;i<param_.num_args*3;++i)
    {
      TYPE_ASSIGN_CHECK(*in_type,i,mshadow::kFloat32)
    }

#else 
    for(size_t i=0;i<param_.num_args*3;++i)
    {
      if(i<4)
      {
        TYPE_ASSIGN_CHECK(*in_type,i,mshadow::kInt8)
      }else
      {
      TYPE_ASSIGN_CHECK(*in_type,i,mshadow::kFloat32)
      }
    }
    
#endif


  TYPE_ASSIGN_CHECK(*out_type,0,(*in_type)[0])

  TYPE_ASSIGN_CHECK(*out_type,1,mshadow::kFloat32)
  TYPE_ASSIGN_CHECK(*out_type,2,mshadow::kFloat32)

  return true;
}

inline static bool QuantizedConcatForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                                 const int dev_mask,
                                                 DispatchMode* dispatch_mode,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
                                          
  CHECK(!in_attrs->empty());
  CHECK_EQ(out_attrs->size(), 3U);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kCSRStorage)
      && param.dim == 0) {
    dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }



#if MXNET_USE_MKLDNN == 1
  if (!dispatched && dev_mask == mshadow::cpu::kDevMask
      && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)
      && param.dim > 0) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
#endif


  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }


#if MXNET_USE_MKLDNN == 1
  if (!MKLDNNEnvSet())
    *dispatch_mode = DispatchMode::kFComputeFallback;
#endif


  for(size_t i=0;i<out_attrs->size();i++)
  {
    (*out_attrs)[i]=kDefaultStorage;
  }
  return true;
}


NNVM_REGISTER_OP(_contrib_quantized_concat)
MXNET_ADD_SPARSE_OP_ALIAS(quantized_concat)
.add_alias("quantized_concat")
.describe(R"code(concat operator for input and output data type of int8.
The input and output data comes with min and max thresholds for quantizing
the float32 data into int8.
.. Note::
    This operator only supports forward propogation. DO NOT use it in training.")code" ADD_FILELINE)
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<bool>("TIsMKLDNN", true)
#endif//CONCAT_FORWARD_ATTRS
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args*3;
}) 
.set_num_outputs(3) 
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  std::vector<std::string> ret;
  for (int i = 0; i < params.num_args; ++i) {
    ret.push_back(std::string("arg") + std::to_string(i));
   
  }
   for (int j = 0; j < params.num_args; ++j) {
    ret.push_back(std::string("arg_min") + std::to_string(j));
    ret.push_back(std::string("arg_max") + std::to_string(j));
  }
  return ret;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuantizedConcatShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedConcatType)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedConcatForwardInferStorageType)
.set_attr<FNeedRequantize>("FNeedRequalsntize",[](const NodeAttrs& attrs){return false;})
// .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
// .add_argument("min_data", "NDArray-or-Symbol[]", "Minimum value of data.")
// .add_argument("max_data", "NDArray-or-Symbol[]", "Maximum value of data.")
.add_arguments(ConcatParam::__FIELDS__());


// _rnn_param_concat is a custom concat op with specialized infer_shape,
// which handles the case where the first one or two inputs may have
// unknown shape that can be inferred from output shape.
// NNVM_REGISTER_OP(_quantized_rnn_param_concat)
// #if MXNET_USE_MKLDNN == 1
// .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
//   return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
// })
// #endif//CONCAT_FORWARD_ATTRS
// .set_attr<nnvm::FInferShape>("FInferShape", QuantizedRNNParamConcatShape)
// .set_attr<nnvm::FInferType>("FInferType", QuantizedConcatType)
// .set_attr<FInferStorageType>("FInferStorageType", QuantizedConcatForwardInferStorageType)
// .add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
// .add_arguments(ConcatParam::__FIELDS__());



NNVM_REGISTER_OP(Concat)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    ConcatParam param;
    param.Init(attrs.dict);
    // TODO(junwu): Uncomment the following line and remove the above lines
    // after pooling op is refactored
    // const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
    nnvm::NodePtr node = nnvm::Node::Create();

    #if MXNET_USE_MKLDNN == 1
    #define USE_QUANTIZEATION true
    #else 
    #define USE_QUANTIZEATION false
    #endif
    if (USE_QUANTIZEATION)// TODO(junming) 
     {

      node->attrs.op = Op::Get("_contrib_quantized_concat");
      node->attrs.name = "quantized_" + attrs.name;
    } else {
      node->attrs.op = Op::Get("Concat");
      node->attrs.name = attrs.name;
    }
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
});




}
}