import logging
import traceback

import tvm
from tvm.contrib.popen_pool import PopenPoolExecutor
from .measure import MeasureErrorNo, MeasureInput, MeasureResult

from extmapper.mem_simulators.hlscnn_mem_sim import HLSCNNConv2DMemSimulator
from extmapper.mem_simulators.flexasr_mem_sim import FlexASRLinearLayerMemSimulator
from extmapper.mem_simulators.vta_mem_sim import VTAGEMMMemSimulator
from extmapper.mem_simulators.hlscnn_single_spad_mem_sim import HLSCNNConv2DMemSimulator_MemAnalysis
from extmapper.hw_models.hlscnn_model import HLSCNNModel as hlscnn
from extmapper.hw_models.hlscnn_model import HLSCNNModel_singleSPAD as hlscnn_single_spad
from extmapper.hw_models.flexasr_model import FlexASRModel as flexasr
from math import ceil

logger = logging.getLogger("autotvm")


def hlscnn_sim(measure_input):
    assert "3la_hlscnn" in measure_input.task.name
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
    if "single_spad" in measure_input.task.name:
        print("running hlscnn single spad...")
        mem_partition = [hlscnn_single_spad.SPAD_SIZE // 3] * 3
        hlscnn_mem_simulator = HLSCNNConv2DMemSimulator_MemAnalysis(
            layer_info=layer_info, schedule=schedule, mem_partition=mem_partition
        )
    else:
        mem_partition = (hlscnn.SPAD1_SIZE // 2, hlscnn.SPAD1_SIZE // 2)
        hlscnn_mem_simulator = HLSCNNConv2DMemSimulator(
            layer_info=layer_info, schedule=schedule, act_mem_part=mem_partition
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


def flexasr_sim(measure_input):
    assert "3la_flexasr" in measure_input.task.name
    N, H, W, CO, CI, KH, KW, stride, padding = measure_input.task.args
    entity = measure_input.config.to_json_dict()["entity"]

    def find_tile_size(name):
        for e in entity:
            if e[0] == name:
                return e[-1][-1]
        raise NameError

    tb = find_tile_size("tile_b")
    ty = find_tile_size("tile_y")
    tx = find_tile_size("tile_x")
    loopOrder = entity[-1][-1]
    loopDim = "tyx"
    loopOrder = tuple(loopDim[i] for i in loopOrder)
    # print(entity)
    # print(tk, tc, th, tw)
    # print(loopOrder)
    out_h = (H + 2 * padding - (KH - 1)) // stride
    out_w = (H + 2 * padding - (KW - 1)) // stride
    batch = out_h * out_w
    in_dim = KH * KW * CI
    out_dim = CO

    layer_info = (batch, in_dim, out_dim)
    loop_bound_dict = {
        "t": ceil(batch / tb),
        "y": ceil(out_dim / ty),
        "x": ceil(in_dim / tx),
    }
    loopBound = tuple(loop_bound_dict[i] for i in loopOrder)
    schedule = (loopOrder, loopBound, tb, tx, ty)
    gb_mem_part = (flexasr.GBCORE_LARGE_BUF_SIZE // 2, flexasr.GBCORE_LARGE_BUF_SIZE // 2)
    flexasr_mem_simualtor = FlexASRLinearLayerMemSimulator(layer_info, schedule, gb_mem_part)
    try:
        result = flexasr_mem_simualtor.fast_sim()
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


def vta_sim(measure_input):
    assert "3la_vta" in measure_input.task.name
    N, H, W, CO, CI, KH, KW, stride, padding = measure_input.task.args
    entity = measure_input.config.to_json_dict()["entity"]

    def find_tile_size(name):
        for e in entity:
            if e[0] == name:
                return e[-1][-1]
        raise NameError

    tb = find_tile_size("tile_b")
    ty = find_tile_size("tile_y")
    tx = find_tile_size("tile_x")
    loopOrder = entity[-1][-1]
    loopDim = "tyx"
    loopOrder = tuple(loopDim[i] for i in loopOrder)
    # print(entity)
    # print(tk, tc, th, tw)
    # print(loopOrder)
    out_h = (H + 2 * padding - (KH - 1)) // stride
    out_w = (H + 2 * padding - (KW - 1)) // stride
    batch = out_h * out_w
    in_dim = KH * KW * CI
    out_dim = CO

    layer_info = (batch, in_dim, out_dim)
    loop_bound_dict = {
        "t": ceil(batch / tb),
        "y": ceil(out_dim / ty),
        "x": ceil(in_dim / tx),
    }
    loopBound = tuple(loop_bound_dict[i] for i in loopOrder)
    schedule = (loopOrder, loopBound, tb, tx, ty)
    vta_mem_simulator = VTAGEMMMemSimulator(layer_info, schedule)
    try:
        result = vta_mem_simulator.fast_sim()
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
                        (
                            tb,
                            ex,
                        ),
                        MeasureErrorNo.RUN_TIMEOUT,
                        -1,
                        -1,
                    )
                )
    return results


def measure_on_flexasr(measure_inputs, n_parallel):
    executor = PopenPoolExecutor(timeout=10 * (n_parallel + 1))
    results = []
    for i in range(0, len(measure_inputs), n_parallel):
        futures = []
        for measure_inp in measure_inputs[i : i + n_parallel]:
            ret = executor.submit(flexasr_sim, measure_inp)
            futures.append(ret)
        for future in futures:
            try:
                res = future.result()
                results.append(res)
            except Exception as ex:
                tb = traceback.format_exc()
                results.append(
                    MeasureResult(
                        (
                            tb,
                            ex,
                        ),
                        MeasureErrorNo.RUN_TIMEOUT,
                        -1,
                        -1,
                    )
                )
    return results


def measure_on_vta(measure_inputs, n_parallel):
    executor = PopenPoolExecutor(timeout=10 * (n_parallel + 1))
    results = []
    for i in range(0, len(measure_inputs), n_parallel):
        futures = []
        for measure_inp in measure_inputs[i : i + n_parallel]:
            ret = executor.submit(vta_sim, measure_inp)
            futures.append(ret)
        for future in futures:
            try:
                res = future.result()
                results.append(res)
            except Exception as ex:
                tb = traceback.format_exc()
                results.append(
                    MeasureResult(
                        (
                            tb,
                            ex,
                        ),
                        MeasureErrorNo.RUN_TIMEOUT,
                        -1,
                        -1,
                    )
                )
    return results
