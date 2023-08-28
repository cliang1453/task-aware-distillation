# coding=utf-8
# Copyright 2020, Microsoft and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TEDDeBERTa-v2 model configuration"""
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config

class TEDDebertaV2Config(DebertaV2Config):

    model_type = "ted-deberta-v2"

    def __init__(
        self,
        train_filters=False,
        filter_interval=1,
        filter_output_dim=None, 
        filter_nonlinear=False,
        filter_disabled=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.train_filters = train_filters
        self.filter_interval = filter_interval
        self.filter_output_dim = filter_output_dim
        self.filter_nonlinear = filter_nonlinear
        self.filter_disabled = filter_disabled