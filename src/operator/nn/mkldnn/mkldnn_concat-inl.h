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
 * \file mkldnn_concat-inl.h
 * \brief
 * \author Wenting Jiang Junming Chen
*/


#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_

#if MXNET_USE_MKLDNN == 1

#include "../concat-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
#include <utility>
#include <mkldnn.hpp>


namespace mxnet {
namespace op {

class MKLDNNConcatFwd {
  std::shared_ptr<mkldnn::concat> fwd;
  std::vector<std::shared_ptr<mkldnn::memory>> data;
  std::vector<mkldnn::primitive::at> data_mem;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::concat::primitive_desc fwd_pd;

  MKLDNNConcatFwd(
      int concat_dim,
      const std::vector<mkldnn::memory::primitive_desc> &data_md): fwd_pd(concat_dim, data_md) {
    data.resize(data_md.size());
  }
  void SetNewMem(const std::vector<const mkldnn::memory *> &in_data,
                 const mkldnn::memory &output);
  const mkldnn::concat &GetFwd() const;
  
};


MKLDNNConcatFwd &GetConcatForward(
    int concat_dim, const std::vector<NDArray> &in_data,
    const std::vector<mkldnn::memory::primitive_desc> &data_md);
void MKLDNNConcatForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                         const std::vector<NDArray> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &out_data);
void MKLDNNConcatBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs);

}
}
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONCAT_INL_H_