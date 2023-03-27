# This file defines the AutoTVM template for HLSCNN Conv2D operator

import tvm
from tvm import te, topi
from tvm import autotvm


@autotvm.template("conv2d_on_3la_hlscnn")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"
    assert padding == 0, "HLSCNN does not support padding"

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])

    n, f, h, w = s[conv].op.axis
    c, kh, kw = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    # define the tile sizes candidates for each axis as a full scan of the search space
    cfg.define_split(
        "tile_f",
        f,
        num_outputs=2,
        policy="candidate",
        candidate=list((-1, i) for i in range(1, CO + 1)),
    )
    cfg.define_split(
        "tile_h",
        h,
        num_outputs=2,
        policy="candidate",
        candidate=list((-1, i) for i in range(1, H + 1)),
    )
    cfg.define_split(
        "tile_w",
        w,
        num_outputs=2,
        policy="candidate",
        candidate=list((-1, i) for i in range(1, W + 1)),
    )
    cfg.define_split(
        "tile_c",
        c,
        num_outputs=2,
        policy="candidate",
        candidate=list((-1, i) for i in range(1, CI + 1)),
    )

    fo, fi = cfg["tile_f"].apply(s, conv, f)
    co, ci = cfg["tile_c"].apply(s, conv, c)
    ho, hi = cfg["tile_h"].apply(s, conv, h)
    wo, wi = cfg["tile_w"].apply(s, conv, w)

    cfg.define_reorder("loop_reorder", [fo, co, ho, wo], "all")

    return s, [data, kernel, conv]
