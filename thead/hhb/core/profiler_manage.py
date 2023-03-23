# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unnecessary-comprehension
"""Manage profiler"""
import collections
import os
import logging
import json

import tvm
from tvm.target import Target

from .common import HHBException


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


def convert_tvm_trace2python(data):
    """convert Array[Map<string, Map<string, objectref>>] into python format

    Parameters
    ----------
    data : Array[Map<string, Map<string, objectref>>]
        original tvm data

    Returns
    -------
    res : list[dict[str, dict[str, object]]]
        result converted into python data
    """
    res = list()
    for inner_map in data:
        inner_map_dict = {}
        for k, v in inner_map.items():
            inner_inner_map_dict = {}
            for kk, vv in v.items():
                if isinstance(vv, tvm.tir.expr.IntImm):
                    inner_inner_map_dict[str(kk)] = int(vv)
                elif isinstance(vv, tvm.runtime.container.String):
                    inner_inner_map_dict[str(kk)] = str(vv)
            inner_map_dict[str(k)] = inner_inner_map_dict
        res.append(inner_map_dict)
    return res


def aitrace_options(indicator, path):
    """Create aitrace options

    Parameters
    ----------
    indicator : list[str]
        What kind of indicator we will profile

    path : str
        The results will be save in this path

    Returns
    -------
    res : Target
        The result target
    """
    if not isinstance(indicator, list):
        raise HHBException("indicator should be list instead of {}".format(type(indicator)))
    target_str = "aitrace"
    target_str += " -type=" + ",".join(indicator)

    if path != "":
        target_str += " -path=" + path

    return Target(target_str)


def get_cal_total_info(data):
    """Get total information of calculation amount from origin data

    Parameters
    ----------
    data : list[dict[str, dict[str, object]]]
        Original data

    res : dict
        Total information
    """
    res = {
        "fused_mul_add": 0,
        "mul": 0,
        "add": 0,
        "sub": 0,
        "exp": 0,
        "comp": 0,
        "div": 0,
    }
    for d in data:
        inner_data = d["calculation_amount"]
        for k, v in inner_data.items():
            res[k] += v
    return res


def get_mem_total_info(data):
    """Get total information of memory from origin data

    Parameters
    ----------
    data : list[dict[str, dict[str, object]]]
        Original data

    res : dict
        Total information
    """
    res = {
        "params": 0,
        "output": 0,
        "accum_ddr": 0,
        "coeff_ddr": 0,
        "input_ddr": 0,
        "output_ddr": 0,
    }
    for d in data:
        inner_data = d["memory"]
        for k, v in inner_data.items():
            if k in res:
                res[k] += v
    return res


def print_cal_total_info(info):
    macc = info["fused_mul_add"]
    flops = 0
    for k, v in info.items():
        if k != "fused_mul_add":
            flops += v
    print(f"Total calculation amount: macc={macc}, flops={flops}")


def print_mem_total_info(info):
    print(
        f"Total memory(float): params={info['params'] * 4} bytes, output={info['output'] * 4} bytes.\n"
        f"Total ddr: accum_ddr={info['accum_ddr']} bytes, coeff_ddr={info['coeff_ddr']} bytes,\n"
        f"           input_ddr={info['input_ddr']} bytes, output_ddr={info['output_ddr']} bytes."
    )


def profile_th1520_trace(trace_data, indicator, frequency):
    """Profile trace data for th1520.

    Parameters
    ----------
    trace_data : dict
        Original trace data in th1520.

    indicator : list
        Indicator type to profile.

    frequency : int
        NPU frequency

    Returns
    -------
    result : list[OrderedDict]
        The results of profiler.
    """
    result = list()
    for layer in trace_data["layers"]:
        l_data = collections.OrderedDict()
        # op
        l_data["op"] = collections.OrderedDict()
        l_data["op"]["type"] = layer["ops"]
        l_data["op"]["name"] = layer["names"]
        # ddr
        if "mem" in indicator or "all" in indicator:
            l_data["memory"] = collections.OrderedDict()
            l_data["memory"]["accum_ddr"] = layer["accum_ddr"]
            l_data["memory"]["coeff_ddr"] = layer["coeff_ddr"]
            l_data["memory"]["input_ddr"] = layer["input_ddr"]
            l_data["memory"]["output_ddr"] = layer["output_ddr"]
        # cycle
        if "cycle" in indicator or "all" in indicator:
            l_data["cycle"] = layer["cycles"]
            l_data["time_ms"] = layer["cycles"] / frequency * 1000  # ms

        result.append(l_data)
    return result


def get_ddr_total_info(data):
    """Get total information of ddr from profile data

    Parameters
    ----------
    data : list[dict[str, dict[str, object]]]
        Original data

    res : dict
        Total information
    """
    res = {"ddr": 0}
    for d in data:
        for k, v in d["ddr"].items():
            res["ddr"] += v
    return res


def print_ddr_total_info(info):
    print(f"Total ddr: {info['ddr']}")


def get_cycle_total_info(data):
    """Get total information of cycle from profile data

    Parameters
    ----------
    data : list[dict[str, dict[str, object]]]
        Original data

    res : dict
        Total information
    """
    res = {"cycle": 0, "time_ms": 0.0}
    for d in data:
        res["cycle"] += d["cycle"]
        res["time_ms"] += d["time_ms"]
    return res


def print_cycle_total_info(info):
    print(f"Total cycle: {info['cycle']}")
    print(f"Total time(ms): {info['time_ms']}")


def dump_profile_result(result, output_type, indicator, ir_type, output_dir=None):
    """Dump profile result according to specifying output_type.

    Parameters
    ----------
    result : list[OrderedDict]
        The results of profiler.

    output_type : list
        How to dump result.

    indicator : list
        Indicator type to profile.

    ir_type : str
        The ir type to profile

    output_dir : str
        The output directory.
    """
    if "json" in output_type or "all" in output_type:
        with open(os.path.join(output_dir, "model_aitrace.json"), "w") as f:
            json.dump(result, f, indent=2)
        logger.info(
            "save model aitrace data into %s", os.path.join(output_dir, "model_aitrace.json")
        )
    if "print" in output_type or "all" in output_type:
        print(result)
    if "total" in output_type or "all" in output_type:
        print("Toal profiler information as follows:")

        if ir_type == "relay":
            if "cal" in indicator or "all" in indicator:
                total_info = get_cal_total_info(result)
                print_cal_total_info(total_info)

            if "mem" in indicator or "all" in indicator:
                total_info = get_mem_total_info(result)
                print_mem_total_info(total_info)
        elif ir_type == "th1520":
            if "mem" in indicator or "all" in indicator:
                total_info = get_mem_total_info(result)
                print_mem_total_info(total_info)

            if "cycle" in indicator or "all" in indicator:
                total_info = get_cycle_total_info(result)
                print_cycle_total_info(total_info)
