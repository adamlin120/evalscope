import sqlite3
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest


def _stub_modelscope():
    if 'modelscope' in sys.modules:
        return

    modelscope_module = types.ModuleType('modelscope')
    modelscope_utils = types.ModuleType('modelscope.utils')
    modelscope_utils_constant = types.ModuleType('modelscope.utils.constant')
    modelscope_utils_constant.DEFAULT_REPOSITORY_REVISION = 'main'

    modelscope_utils_file_utils = types.ModuleType('modelscope.utils.file_utils')

    def _dummy_path():
        return ''

    modelscope_utils_file_utils.get_dataset_cache_root = _dummy_path
    modelscope_utils_file_utils.get_model_cache_root = _dummy_path

    modelscope_module.utils = modelscope_utils
    modelscope_utils.constant = modelscope_utils_constant
    modelscope_utils.file_utils = modelscope_utils_file_utils

    sys.modules['modelscope'] = modelscope_module
    sys.modules['modelscope.utils'] = modelscope_utils
    sys.modules['modelscope.utils.constant'] = modelscope_utils_constant
    sys.modules['modelscope.utils.file_utils'] = modelscope_utils_file_utils


_stub_modelscope()


def _stub_evalscope_package():
    if 'evalscope' in sys.modules:
        return

    package = types.ModuleType('evalscope')
    package_dir = Path(__file__).resolve().parents[3] / 'evalscope'
    package.__path__ = [str(package_dir)]
    package.__package__ = 'evalscope'
    package.__spec__ = ModuleSpec('evalscope', loader=None, is_package=True)
    sys.modules['evalscope'] = package


_stub_evalscope_package()

from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.perf.utils.db_util import (
    DEFAULT_PERCENTILES,
    PercentileMetrics,
    create_result_table,
    get_percentile_results,
    insert_benchmark_data,
)


def _insert_success_row(cursor, idx, *, ttft, latency, prompt_tokens, completion_tokens):
    data = BenchmarkData(
        request={'idx': idx},
        start_time=0.0,
        completed_time=latency,
        chunk_times=[ttft, latency],
        success=True,
        response_messages=[{'message': 'ok'}],
    )
    data.query_latency = latency
    data.first_chunk_latency = ttft
    data.prompt_tokens = prompt_tokens
    data.completion_tokens = completion_tokens
    data.max_gpu_memory_cost = 0.0
    data.time_per_output_token = latency / completion_tokens if completion_tokens else 0.0
    insert_benchmark_data(cursor, data)


def test_get_percentile_results_supports_sort_metric(tmp_path):
    db_path = tmp_path / 'results.db'
    with sqlite3.connect(db_path) as con:
        cursor = con.cursor()
        create_result_table(cursor)

        samples = [
            {'ttft': 1.0, 'latency': 30.0, 'prompt_tokens': 10, 'completion_tokens': 10},
            {'ttft': 2.0, 'latency': 10.0, 'prompt_tokens': 20, 'completion_tokens': 20},
            {'ttft': 3.0, 'latency': 20.0, 'prompt_tokens': 30, 'completion_tokens': 30},
        ]

        for idx, sample in enumerate(samples):
            _insert_success_row(cursor, idx, **sample)

        con.commit()

    default_result = get_percentile_results(str(db_path))
    sorted_result = get_percentile_results(str(db_path), 'ttft')

    assert default_result[PercentileMetrics.PERCENTILES] == [f'{p}%' for p in DEFAULT_PERCENTILES]
    assert sorted_result[PercentileMetrics.PERCENTILES] == [f'{p}%' for p in DEFAULT_PERCENTILES]

    # Default behaviour calculates the percentile per metric independently.
    assert default_result[PercentileMetrics.LATENCY][0] == pytest.approx(10.0)

    # Sorted behaviour aligns all metrics by the TTFT order.
    assert sorted_result[PercentileMetrics.LATENCY][0] == pytest.approx(30.0)
    assert sorted_result[PercentileMetrics.TTFT][0] == pytest.approx(1.0)


def test_get_percentile_results_invalid_metric(tmp_path):
    db_path = tmp_path / 'results_invalid.db'
    with sqlite3.connect(db_path) as con:
        cursor = con.cursor()
        create_result_table(cursor)
        _insert_success_row(cursor, 0, ttft=1.0, latency=5.0, prompt_tokens=5, completion_tokens=5)
        con.commit()

    with pytest.raises(ValueError):
        get_percentile_results(str(db_path), 'unknown')
