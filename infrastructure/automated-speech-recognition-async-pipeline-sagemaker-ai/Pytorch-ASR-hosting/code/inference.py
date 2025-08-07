# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import io
import sys
import time
import json
import logging
import torch
import boto3
import tempfile
from botocore.exceptions import NoCredentialsError
import nemo.collections.asr as nemo_asr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):
    """
    Load and return the model
    """
    model = nemo_asr.models.ASRModel.restore_from(restore_path="parakeet-tdt-0.6b-v2.nemo").to(DEVICE)
    
    # Enable local attention
    model.change_attention_model("rel_pos_local_attn", [128, 128])  # local attn
     
    # Enable chunking for subsampling module
    model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
    print(f'parakeet model has been loaded to this device: {model.device.type}')
    return model

def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
    """
    Transform the input data and generate a transcription result
    """
    logging.info("Check out the request_body type: %s", type(request_body))
    start_time = time.time()
    
    file = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(file.read())



    logging.info("Start generating the transcription ...")
    result = model.transcribe([tfile.name])
    logging.info("Transcription generation completed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Elapsed time: %s seconds", elapsed_time)
    if hasattr(result[0], 'text'):
        text_result = [hyp.text for hyp in result]
    else:
        text_result = result
    return json.dumps(text_result), response_content_type
