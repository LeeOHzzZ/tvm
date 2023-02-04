# This file follows the how-to template from tvm and try to use AutoTVM
# to optimize Conv2D schedule on HLSCNN

import logging
import os
import sys
import numpy as np
from math import ceil

import tvm
from tvm import te, topi
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

from tvm import autotvm

@autotvm.template("conv2d_on_cpu")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"
    assert padding == 0, "HLSCNN does not support padding"

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])
    print(s)
    print(s[conv].op.axis)
    print(s[conv].op.reduce_axis)

    n, f, h, w = s[conv].op.axis
    c, kh, kw = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=2, policy="verbose")
    cfg.define_split("tile_h", h, num_outputs=2, policy="verbose")
    cfg.define_split("tile_w", w, num_outputs=2, policy="verbose")
    cfg.define_split("tile_c", c, num_outputs=2, policy="verbose")

    fo, fi = cfg["tile_f"].apply(s, conv, f)
    co, ci = cfg["tile_c"].apply(s, conv, c)
    ho, hi = cfg["tile_h"].apply(s, conv, h)
    wo, wi = cfg["tile_w"].apply(s, conv, w)

    cfg.define_reorder("loop_reorder", [fo, co, ho, wo], "all")

    return s, [data, kernel, conv]


@autotvm.template("conv2d_on_hlscnn")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"
    assert padding == 0, "HLSCNN does not support padding"

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])
    print(s)
    print(s[conv].op.axis)
    print(s[conv].op.reduce_axis)

    n, f, h, w = s[conv].op.axis
    c, kh, kw = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=2, policy="verbose")
    cfg.define_split("tile_h", h, num_outputs=2, policy="verbose")
    cfg.define_split("tile_w", w, num_outputs=2, policy="verbose")
    cfg.define_split("tile_c", c, num_outputs=2, policy="verbose")

    fo, fi = cfg["tile_f"].apply(s, conv, f)
    co, ci = cfg["tile_c"].apply(s, conv, c)
    ho, hi = cfg["tile_h"].apply(s, conv, h)
    wo, wi = cfg["tile_w"].apply(s, conv, w)

    cfg.define_reorder("loop_reorder", [fo, co, ho, wo], "all")

    return s, [data, kernel, conv]



if __name__ == "__main__":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 226, 226, 64, 8, 3, 3, 1, 0
    task = autotvm.task.create(
        "conv2d_on_cpu", args=(N, H, W, CO, CI, KH, KW, stride, padding), target="llvm"
    )
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=1))

    if os.path.exists("hlscnn_conv2d.log"):
        os.remove("hlscnn_conv2d.log")
    # tuner = autotvm.tuner.RandomTuner(task)
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=100,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("hlscnn_conv2d.log")],
    )
    dispatch_context = autotvm.apply_history_best("hlscnn_conv2d.log")
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest Config:", best_config)
    entity = best_config.to_json_dict()["entity"]
    def find_tile_size(name):
        for e in entity:
            if e[0] == name:
                return e[-1][-1]
        raise NameError
    tk = find_tile_size("tile_f")
    tc = find_tile_size("tile_c")
    th = find_tile_size("tile_h")
    tw = find_tile_size("tile_w")
    loopOrder = entity[-1][-1]
    loopDim = "kchw"
    loopOrder = tuple(loopDim[i] for i in loopOrder)
    loop_bound_dict = {
        "c": ceil(CI / tc),
        "k": ceil(CO / tk),
        "h": ceil(H / th),
        "w": ceil(W / tw),
    }
    print(loop_bound_dict, loopOrder)