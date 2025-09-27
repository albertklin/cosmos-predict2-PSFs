# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
from collections import OrderedDict
from contextlib import contextmanager

import torch
from safetensors.torch import load as safetensors_torch_load

from imaginaire.utils import log
from imaginaire.utils.easy_io import easy_io


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):  # noqa: B008
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in ["safetensors", "bin", "ckpt", "pth", "pt"]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    fake_checkpoint = bool(os.environ.get("COSMOS_PREDICT2_FAKE_CHECKPOINT"))
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(
            file_path,
            torch_dtype=torch_dtype,
            fake_checkpoint=fake_checkpoint,
        )
    else:
        return load_state_dict_from_bin(
            file_path,
            torch_dtype=torch_dtype,
            fake_checkpoint=fake_checkpoint,
        )


def load_state_dict_from_safetensors(file_path, torch_dtype=None, fake_checkpoint: bool = False):
    backend_args = None
    if fake_checkpoint:
        log.warning(
            "COSMOS_PREDICT2_FAKE_CHECKPOINT is set; using safetensors metadata from '%s' without loading weights.",
            file_path,
        )
        return _build_fake_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)

    byte_stream = easy_io.load(file_path, backend_args=backend_args, file_format="byte")
    state_dict = safetensors_torch_load(byte_stream)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, fake_checkpoint: bool = False):
    backend_args = None
    map_location = "meta" if fake_checkpoint else "cpu"
    if fake_checkpoint:
        log.warning(
            "COSMOS_PREDICT2_FAKE_CHECKPOINT is set; loading metadata for '%s' on the meta device.",
            file_path,
        )
    state_dict = easy_io.load(
        file_path,
        backend_args=backend_args,
        file_format="pt",
        map_location=map_location,
        weights_only=False,
    )
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def _build_fake_state_dict_from_safetensors(file_path: str, torch_dtype=None):
    try:
        with open(file_path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size).decode("utf-8"))
    except (OSError, ValueError):
        byte_stream = easy_io.load(file_path, backend_args=None, file_format="byte")
        header_size = int.from_bytes(byte_stream[:8], "little")
        header = json.loads(byte_stream[8 : 8 + header_size].decode("utf-8"))
    except json.JSONDecodeError as exc:
        log.error(
            "Failed to decode safetensors metadata from '%s' under fake checkpoint mode: %s", file_path, exc
        )
        return {}

    tensors = header.get("tensors", {})
    fake_state = OrderedDict()
    for name, tensor_info in tensors.items():
        dtype_key = tensor_info.get("dtype")
        shape = tuple(tensor_info.get("shape", ()))
        dtype = _SAFE_TORCH_DTYPES.get(dtype_key, torch.float32)
        if dtype_key not in _SAFE_TORCH_DTYPES:
            log.warning(
                "Unknown safetensors dtype '%s' for tensor '%s' in '%s'; defaulting to torch.float32.",
                dtype_key,
                name,
                file_path,
            )
        if torch_dtype is not None:
            dtype = torch_dtype
        fake_state[name] = torch.empty(shape, device="meta", dtype=dtype)
    return fake_state


_SAFE_TORCH_DTYPES = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape) >= 1 and source_state_dict[name].shape[0] % 3 == 0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(
                        search_parameter(source_state_dict[name][i * length : i * length + length], target_state_dict)
                    )
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():  # noqa: B007
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts


def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()
