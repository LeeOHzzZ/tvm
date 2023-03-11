# This file defines the AutoTVM template for HLSCNN Conv2D operator

import tvm
from tvm import te, topi
from tvm import autotvm


@autotvm.template("conv2d_on_3la_flexasr_via_im2col")
def conv2d_no_batching_im2col(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    out_h = (H + 2 * padding - (KH - 1)) // stride
    out_w = (H + 2 * padding - (KW - 1)) // stride
    batch = out_h * out_w
    in_dim = KH * KW * CI
    out_dim = CO

    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    dense = topi.nn.dense(data, weight)
    s = te.create_schedule([dense.op])

    b, y = s[dense].op.axis
    x = s[dense].op.reduce_axis[0]

    cfg = autotvm.get_config()
    cfg.define_split("tile_b", b, num_outputs=2, policy="verbose")
    cfg.define_split("tile_y", y, num_outputs=2, policy="verbose")
    cfg.define_split("tile_x", x, num_outputs=2, policy="verbose")

    bo, bi = cfg["tile_b"].apply(s, dense, b)
    yo, yi = cfg["tile_y"].apply(s, dense, y)
    xo, xi = cfg["tile_x"].apply(s, dense, x)

    cfg.define_reorder("loop_reorder", [bo, yo, xo], "all")

    return s, [data, weight, dense]
