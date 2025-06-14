#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("phone")
@dataclass
class PhoneTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = False
    x_step_size = y_step_size = z_step_size = 0.01 # FIX: arbitrary value
    port = 5010 # FIX: arbitrary value for the phone HTTP server
    

@TeleoperatorConfig.register_subclass("camera")
@dataclass
class CameraTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = False
    port: int = 11310 
    

