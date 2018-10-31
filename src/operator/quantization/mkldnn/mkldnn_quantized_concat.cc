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
 * \file mkldnn_quantized_concat.cc
 * \brief
 * \author Junming Chen
*/

#if MXNET_USE_MKLDNN == 1
#include "../../nn/mkldnn/mkldnn_concat-inl.h"
#include<iostream>

namespace mxnet {
namespace op {

// static void scaleData(NDArray &input,float factor)
// {
  
//   input*=factor;
//   using namespace std;
//   cout<<input.shape()<<endl;

//   // unsigned char*  data_ptr=&input;
//   // const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
//   // #pragma omp parallel for num_threads(omp_threads)
//   // for (size_t i = 0; i < 10; ++i) {
//   //   (*data_ptr)=((*data_ptr)*factor);
//   //   data_ptr++;
//   // }
// }


static void scaleData(NDArray &input,float factor)
{
 
  using namespace std;

  auto newinput=input.Reorder2Default();
  unsigned char*  data_ptr=newinput.data().dptr<unsigned char>();
  const TShape& ishape =input.shape();
  size_t size=1;
  for(size_t j =0;j<ishape.ndim();j++)
  {
      size*=ishape[j];
  }

  #pragma omp parallel for num_threads(mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (size_t i = 0; i < size; ++i) {
    data_ptr[i]=(unsigned char)(int)(data_ptr[i]*factor);
  }
  CopyFromToCsrImpl(newinput,input,input.ctx());
}

static void MKLDNNQuantizedConcatForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                          const std::vector<NDArray> &in_data,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &out_data) {
                 
 
  CHECK(in_data[0].dtype() == mshadow::kUint8
    || in_data[0].dtype() == mshadow::kInt8)
    << "mkldnn_quantized_concat op only supports uint8 and int8 as input type";
  TmpMemMgr::Get()->Init(ctx.requested[concat_enum::kTempSpace]);
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int num_in_data = param.num_args;
  int concat_dim = param.dim;
  std::vector<mkldnn::memory::primitive_desc> data_md;
  std::vector<const mkldnn::memory *> data_mem;
  std::vector<NDArray> new_in_data;
  data_md.reserve(num_in_data);
  data_mem.reserve(num_in_data);
  for (int i = 0; i < num_in_data; i++) {
    const mkldnn::memory *tmp_mem = in_data[i].GetMKLDNNData();
    mkldnn::memory::primitive_desc tmp_pd = tmp_mem->get_primitive_desc();
    data_md.push_back(tmp_pd);
    data_mem.push_back(tmp_mem);
    new_in_data.push_back(in_data[i]);
  }
  

  
  //coordinate the scale of diff input
  int maxIndex=-1;
  float maxNum=-1.;
  for (int j = num_in_data+1; j < num_in_data*3; j+=2) 
  {
    
     if(in_data[j].data().dptr<float>()[0]>maxNum)
     {
       maxNum=in_data[j].data().dptr<float>()[0];
       maxIndex=(j-num_in_data)/2;
     }
  }//maxIndex*2+num_in_data+1

  //scale
  // #pragma omp parallel for num_threads(mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int k = 0; k < num_in_data; k++) {
     if(k!=maxIndex)
     {
       float factor=in_data[k*2+num_in_data+1].data().dptr<float>()[0]/maxNum;
      scaleData(new_in_data[k],factor);
      // new_in_data[k]*=factor;
     }
  }


  MKLDNNConcatFwd &fwd = GetConcatForward(concat_dim, new_in_data, data_md);
  mxnet::mkldnn_output_t out_mem = CreateMKLDNNMem(out_data[concat_enum::kOut],
                                                   fwd.fwd_pd.dst_primitive_desc(),
                                                   req[concat_enum::kOut]);


  out_data[1].data().dptr<float>()[0]=in_data[maxIndex*2+num_in_data].data().dptr<float>()[0];
  out_data[2].data().dptr<float>()[0]=in_data[maxIndex*2+num_in_data+1].data().dptr<float>()[0];

  fwd.SetNewMem(data_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
  CommitOutput(out_data[concat_enum::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}




NNVM_REGISTER_OP(_contrib_quantized_concat)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedConcatForward);

// NNVM_REGISTER_OP(_quantized_rnn_param_concat)
// .set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedConcatForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1