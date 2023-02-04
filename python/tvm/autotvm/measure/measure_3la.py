import logging
import traceback

import tvm
from tvm.contrib.popen_pool import PopenPoolExecutor
from .measure import MeasureErrorNo, MeasureInput, MeasureResult

from extmapper.mem_simulators.hlscnn_mem_sim import HLSCNNConv2DMemSimulator
from math import ceil

logger = logging.getLogger("autotvm")

def hlscnn_sim(measure_input):
    assert "hlscnn" in measure_input.task.name
    N, H, W, CO, CI, KH, KW, stride, padding = measure_input.task.args
    entity = measure_input.config.to_json_dict()["entity"]
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
    # print(entity)
    # print(tk, tc, th, tw)
    # print(loopOrder)
    layer_info = (CI, CO, H, W, KH, KW, stride, padding)
    loop_bound_dict = {
        "c": ceil(CI / tc),
        "k": ceil(CO / tk),
        "h": ceil(H / th),
        "w": ceil(W / tw),
    }
    loopBound = tuple(loop_bound_dict[i] for i in loopOrder)
    schedule = (loopOrder, loopBound, tc, tk, th, tw)
    hlscnn_mem_simulator = HLSCNNConv2DMemSimulator(
        layer_info=layer_info, schedule=schedule
    )
    try:
        # result = hlscnn_mem_simulator.run()
        result = hlscnn_mem_simulator.fast_sim()
        return MeasureResult(
            costs=(result["total_data_mov"],), 
            error_no=MeasureErrorNo.NO_ERROR,
            all_cost=-1,
            timestamp=-1,
        )
    except:
        return MeasureResult(
            costs=("error", MeasureErrorNo.COMPILE_DEVICE),
            error_no=MeasureErrorNo.COMPILE_DEVICE,
            all_cost=-1,
            timestamp=-1,
        )

def measure_on_hlscnn(measure_inputs, n_parallel):
    executor = PopenPoolExecutor(timeout=10 * (n_parallel + 1))
    results = []
    for i in range(0, len(measure_inputs), n_parallel):
        futures = []
        for measure_inp in measure_inputs[i : i + n_parallel]:
            ret = executor.submit(hlscnn_sim, measure_inp)
            futures.append(ret)
        for future in futures:
            try:
                res = future.result()
                results.append(res)
            except Exception as ex:
                tb = traceback.format_exc()
                results.append(
                    MeasureResult(
                        (tb, ex,),
                        MeasureErrorNo.RUN_TIMEOUT,
                        -1,
                        -1,
                    )
                )
    return results