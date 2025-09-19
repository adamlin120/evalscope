import base64
import json
import math
import os
import pickle
import re
import sqlite3
import sys
from datetime import datetime
from tabulate import tabulate
from typing import Dict, List, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics
from evalscope.utils.logger import get_logger

logger = get_logger()


class DatabaseColumns:
    REQUEST = 'request'
    START_TIME = 'start_time'
    CHUNK_TIMES = 'chunk_times'
    SUCCESS = 'success'
    RESPONSE_MESSAGES = 'response_messages'
    COMPLETED_TIME = 'completed_time'
    LATENCY = 'latency'
    FIRST_CHUNK_LATENCY = 'first_chunk_latency'
    PROMPT_TOKENS = 'prompt_tokens'
    COMPLETION_TOKENS = 'completion_tokens'
    MAX_GPU_MEMORY_COST = 'max_gpu_memory_cost'
    TIME_PER_OUTPUT_TOKEN = 'time_per_output_token'


def load_prompt(prompt_path_or_text):
    if prompt_path_or_text.startswith('@'):
        with open(prompt_path_or_text[1:], 'r', encoding='utf-8') as file:
            return file.read()
    return prompt_path_or_text


def encode_data(data) -> str:
    """Encodes data using base64 and pickle."""
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')


def write_json_file(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def transpose_results(data):
    headers = data.keys()
    rows = zip(*data.values())

    return [dict(zip(headers, row)) for row in rows]


def create_result_table(cursor):
    cursor.execute(
        f'''CREATE TABLE IF NOT EXISTS result(
                      {DatabaseColumns.REQUEST} TEXT,
                      {DatabaseColumns.START_TIME} REAL,
                      {DatabaseColumns.CHUNK_TIMES} TEXT,
                      {DatabaseColumns.SUCCESS} INTEGER,
                      {DatabaseColumns.RESPONSE_MESSAGES} TEXT,
                      {DatabaseColumns.COMPLETED_TIME} REAL,
                      {DatabaseColumns.LATENCY} REAL,
                      {DatabaseColumns.FIRST_CHUNK_LATENCY} REAL,
                      {DatabaseColumns.PROMPT_TOKENS} INTEGER,
                      {DatabaseColumns.COMPLETION_TOKENS} INTEGER,
                      {DatabaseColumns.MAX_GPU_MEMORY_COST} REAL,
                      {DatabaseColumns.TIME_PER_OUTPUT_TOKEN} REAL
                   )'''
    )


def insert_benchmark_data(cursor: sqlite3.Cursor, benchmark_data: BenchmarkData):
    request = encode_data(benchmark_data.request)
    chunk_times = json.dumps(benchmark_data.chunk_times)
    response_messages = encode_data(benchmark_data.response_messages)

    # Columns common to both success and failure cases
    common_columns = (
        request,
        benchmark_data.start_time,
        chunk_times,
        benchmark_data.success,
        response_messages,
        benchmark_data.completed_time,
    )

    if benchmark_data.success:
        # Add additional columns for success case
        additional_columns = (
            benchmark_data.query_latency, benchmark_data.first_chunk_latency, benchmark_data.prompt_tokens,
            benchmark_data.completion_tokens, benchmark_data.max_gpu_memory_cost, benchmark_data.time_per_output_token
        )
        query = f"""INSERT INTO result(
                      {DatabaseColumns.REQUEST}, {DatabaseColumns.START_TIME}, {DatabaseColumns.CHUNK_TIMES},
                      {DatabaseColumns.SUCCESS}, {DatabaseColumns.RESPONSE_MESSAGES}, {DatabaseColumns.COMPLETED_TIME},
                      {DatabaseColumns.LATENCY}, {DatabaseColumns.FIRST_CHUNK_LATENCY}, {DatabaseColumns.PROMPT_TOKENS},
                      {DatabaseColumns.COMPLETION_TOKENS}, {DatabaseColumns.MAX_GPU_MEMORY_COST},
                      {DatabaseColumns.TIME_PER_OUTPUT_TOKEN}
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns + additional_columns)
    else:
        query = f"""INSERT INTO result(
                      {DatabaseColumns.REQUEST}, {DatabaseColumns.START_TIME}, {DatabaseColumns.CHUNK_TIMES},
                      {DatabaseColumns.SUCCESS}, {DatabaseColumns.RESPONSE_MESSAGES}, {DatabaseColumns.COMPLETED_TIME}
                   ) VALUES (?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns)


def get_output_path(args: Arguments) -> str:
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.outputs_dir, current_time, f'{args.name or args.model_id}')
    # Filter illegal characters
    output_path = re.sub(r'[<>:"|?*]', '_', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger.info(f'Save the result to: {output_path}')
    return output_path


def get_result_db_path(args: Arguments):
    result_db_path = os.path.join(args.outputs_dir, 'benchmark_data.db')

    logger.info(f'Save the data base to: {result_db_path}')
    if os.path.exists(result_db_path):
        logger.error(f'The db file {result_db_path} exists, delete it and start again!.')
        sys.exit(1)

    return result_db_path


class PercentileMetrics:
    TTFT = 'TTFT (s)'
    ITL = 'ITL (s)'
    TPOT = 'TPOT (s)'
    LATENCY = 'Latency (s)'
    INPUT_TOKENS = 'Input tokens'
    OUTPUT_TOKENS = 'Output tokens'
    OUTPUT_THROUGHPUT = 'Output (tok/s)'
    TOTAL_THROUGHPUT = 'Total (tok/s)'
    PERCENTILES = 'Percentiles'


DEFAULT_PERCENTILES = [10, 25, 50, 66, 75, 80, 90, 95, 98, 99]

PERCENTILE_METRIC_ORDER = [
    PercentileMetrics.TTFT,
    PercentileMetrics.ITL,
    PercentileMetrics.TPOT,
    PercentileMetrics.LATENCY,
    PercentileMetrics.INPUT_TOKENS,
    PercentileMetrics.OUTPUT_TOKENS,
    PercentileMetrics.OUTPUT_THROUGHPUT,
    PercentileMetrics.TOTAL_THROUGHPUT,
]

_PERCENTILE_METRIC_ALIASES = {
    PercentileMetrics.TTFT: ['ttft', 'time_to_first_token', 'first_chunk_latency', 'first_token_latency'],
    PercentileMetrics.ITL: ['itl', 'inter_token_latency', 'inter_token_latencies'],
    PercentileMetrics.TPOT: ['tpot', 'time_per_output_token', 'per_output_token_latency'],
    PercentileMetrics.LATENCY: ['latency', 'total_latency', 'e2e_latency'],
    PercentileMetrics.INPUT_TOKENS: ['input_tokens', 'prompt_tokens'],
    PercentileMetrics.OUTPUT_TOKENS: ['output_tokens', 'completion_tokens'],
    PercentileMetrics.OUTPUT_THROUGHPUT: ['output_throughput', 'output_tokps', 'output_tokens_per_sec'],
    PercentileMetrics.TOTAL_THROUGHPUT: ['total_throughput', 'total_tokps', 'total_tokens_per_sec'],
}


def _sanitize_metric_alias(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _build_metric_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, aliases in _PERCENTILE_METRIC_ALIASES.items():
        lookup[_sanitize_metric_alias(canonical)] = canonical
        for alias in aliases:
            lookup[_sanitize_metric_alias(alias)] = canonical
    return lookup


PERCENTILE_METRIC_LOOKUP = _build_metric_lookup()


def _resolve_sort_metric(sort_metric: str) -> str:
    sanitized = _sanitize_metric_alias(sort_metric)
    if sanitized not in PERCENTILE_METRIC_LOOKUP:
        raise ValueError(
            f'Unsupported percentile sort metric "{sort_metric}". '
            f'Available metrics: {sorted(set(PERCENTILE_METRIC_LOOKUP.values()))}'
        )
    return PERCENTILE_METRIC_LOOKUP[sanitized]


def inter_token_latencies_from_json(chunk_times_json: str) -> List[float]:
    try:
        chunk_times = json.loads(chunk_times_json)
        return [t2 - t1 for t1, t2 in zip(chunk_times[:-1], chunk_times[1:])]
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error(f'Error parsing chunk times: {exc}')
        return []


def _is_nan(value: Optional[float]) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _round_metric_value(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, 4)
    return value


def _build_row_metrics(row, col_indices) -> Dict[str, Optional[float]]:
    chunk_times_json = row[col_indices[DatabaseColumns.CHUNK_TIMES]]
    chunk_itls = inter_token_latencies_from_json(chunk_times_json)

    latency_raw = row[col_indices[DatabaseColumns.LATENCY]]
    latency = float(latency_raw) if latency_raw is not None else float('nan')

    first_chunk_latency_raw = row[col_indices[DatabaseColumns.FIRST_CHUNK_LATENCY]]
    first_chunk_latency = (
        float(first_chunk_latency_raw) if first_chunk_latency_raw is not None else float('nan')
    )

    prompt_tokens_raw = row[col_indices[DatabaseColumns.PROMPT_TOKENS]]
    prompt_tokens = int(prompt_tokens_raw) if prompt_tokens_raw is not None else 0

    completion_tokens_raw = row[col_indices[DatabaseColumns.COMPLETION_TOKENS]]
    completion_tokens = int(completion_tokens_raw) if completion_tokens_raw is not None else 0

    time_per_output_token_raw = row[col_indices[DatabaseColumns.TIME_PER_OUTPUT_TOKEN]]
    time_per_output_token = (
        float(time_per_output_token_raw) if time_per_output_token_raw is not None else float('nan')
    )

    output_throughput = (
        (completion_tokens / latency) if latency_raw is not None and latency > 0 else float('nan')
    )
    total_throughput = (
        ((prompt_tokens + completion_tokens) / latency)
        if latency_raw is not None and latency > 0 else float('nan')
    )

    average_itl = (sum(chunk_itls) / len(chunk_itls)) if chunk_itls else float('nan')

    return {
        PercentileMetrics.TTFT: first_chunk_latency,
        PercentileMetrics.ITL: average_itl,
        PercentileMetrics.TPOT: time_per_output_token,
        PercentileMetrics.LATENCY: latency,
        PercentileMetrics.INPUT_TOKENS: prompt_tokens,
        PercentileMetrics.OUTPUT_TOKENS: completion_tokens,
        PercentileMetrics.OUTPUT_THROUGHPUT: output_throughput,
        PercentileMetrics.TOTAL_THROUGHPUT: total_throughput,
    }


def _calculate_sorted_percentiles(rows, col_indices, sort_metric: str,
                                  percentiles: List[int]) -> Dict[str, List[float]]:
    row_metrics = [_build_row_metrics(row, col_indices) for row in rows]
    if not row_metrics:
        return _empty_percentile_result(percentiles)

    def sort_key(metric_row):
        value = metric_row.get(sort_metric)
        if value is None or _is_nan(value):
            return (1, float('inf'))
        return (0, value)

    sorted_rows = sorted(row_metrics, key=sort_key)
    if not sorted_rows:
        return {}

    results: Dict[str, List[float]] = {
        PercentileMetrics.PERCENTILES: [f'{p}%' for p in percentiles]
    }

    for metric in PERCENTILE_METRIC_ORDER:
        values: List[float] = []
        for percentile in percentiles:
            if not sorted_rows:
                values.append(float('nan'))
                continue
            idx = int(len(sorted_rows) * percentile / 100)
            if idx >= len(sorted_rows):
                idx = len(sorted_rows) - 1
            selected_value = sorted_rows[idx].get(metric, float('nan'))
            values.append(_round_metric_value(selected_value))
        results[metric] = values

    return results


def _calculate_standard_percentiles(rows, col_indices, percentiles: List[int]) -> Dict[str, List[float]]:
    inter_token_latencies_all: List[float] = []
    for row in rows:
        inter_token_latencies_all.extend(
            inter_token_latencies_from_json(row[col_indices[DatabaseColumns.CHUNK_TIMES]])
        )

    metrics = {
        PercentileMetrics.TTFT: [row[col_indices[DatabaseColumns.FIRST_CHUNK_LATENCY]] for row in rows],
        PercentileMetrics.ITL: inter_token_latencies_all,
        PercentileMetrics.TPOT: [row[col_indices[DatabaseColumns.TIME_PER_OUTPUT_TOKEN]] for row in rows],
        PercentileMetrics.LATENCY: [row[col_indices[DatabaseColumns.LATENCY]] for row in rows],
        PercentileMetrics.INPUT_TOKENS: [row[col_indices[DatabaseColumns.PROMPT_TOKENS]] for row in rows],
        PercentileMetrics.OUTPUT_TOKENS: [row[col_indices[DatabaseColumns.COMPLETION_TOKENS]] for row in rows],
        PercentileMetrics.OUTPUT_THROUGHPUT: [
            (row[col_indices[DatabaseColumns.COMPLETION_TOKENS]] / row[col_indices[DatabaseColumns.LATENCY]])
            if row[col_indices[DatabaseColumns.LATENCY]] > 0 else float('nan') for row in rows
        ],
        PercentileMetrics.TOTAL_THROUGHPUT: [
            ((row[col_indices[DatabaseColumns.PROMPT_TOKENS]] + row[col_indices[DatabaseColumns.COMPLETION_TOKENS]])
             / row[col_indices[DatabaseColumns.LATENCY]])
            if row[col_indices[DatabaseColumns.LATENCY]] > 0 else float('nan') for row in rows
        ]
    }

    results = {PercentileMetrics.PERCENTILES: [f'{p}%' for p in percentiles]}
    for metric_name, data in metrics.items():
        metric_percentiles = calculate_percentiles(data, percentiles)
        results[metric_name] = [metric_percentiles[p] for p in percentiles]

    return results


def _empty_percentile_result(percentiles: List[int]) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {
        PercentileMetrics.PERCENTILES: [f'{p}%' for p in percentiles]
    }
    empties = [float('nan')] * len(percentiles)
    for metric in PERCENTILE_METRIC_ORDER:
        result[metric] = list(empties)
    return result


def calculate_percentiles(data: List[float], percentiles: List[int]) -> Dict[int, float]:
    """
    Calculate the percentiles for a specific list of data.

    :param data: List of values for a specific metric.
    :param percentiles: List of percentiles to calculate.
    :return: Dictionary of calculated percentiles.
    """
    results = {}
    n_success_queries = len(data)
    data.sort()
    for percentile in percentiles:
        try:
            idx = int(n_success_queries * percentile / 100)
            value = data[idx] if data[idx] is not None else float('nan')
            results[percentile] = round(value, 4)
        except IndexError:
            results[percentile] = float('nan')
    return results


def get_percentile_results(result_db_path: str, sort_metric: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Compute and return quantiles for various metrics from the database results.

    :param result_db_path: Path to the SQLite database file.
    :param sort_metric: Optional metric name to align all percentile rows against.
    :return: Dictionary of percentiles for various metrics.
    """

    query_sql = f'''SELECT {DatabaseColumns.START_TIME}, {DatabaseColumns.CHUNK_TIMES}, {DatabaseColumns.SUCCESS},
                    {DatabaseColumns.COMPLETED_TIME}, {DatabaseColumns.LATENCY}, {DatabaseColumns.FIRST_CHUNK_LATENCY},
                    {DatabaseColumns.PROMPT_TOKENS},
                    {DatabaseColumns.COMPLETION_TOKENS}, {DatabaseColumns.TIME_PER_OUTPUT_TOKEN}
                    FROM result WHERE {DatabaseColumns.SUCCESS}=1'''

    percentiles = DEFAULT_PERCENTILES

    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        cursor.execute(query_sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

    if not rows:
        return _empty_percentile_result(percentiles)

    # Create column index mapping
    col_indices = {col: idx for idx, col in enumerate(columns)}

    if sort_metric:
        canonical_metric = _resolve_sort_metric(sort_metric)
        return _calculate_sorted_percentiles(rows, col_indices, canonical_metric, percentiles)

    return _calculate_standard_percentiles(rows, col_indices, percentiles)


def summary_result(args: Arguments, metrics: BenchmarkMetrics, result_db_path: str) -> Tuple[Dict, Dict]:
    result_path = os.path.dirname(result_db_path)
    write_json_file(args.to_dict(), os.path.join(result_path, 'benchmark_args.json'))

    metrics_result = metrics.create_message()
    write_json_file(metrics_result, os.path.join(result_path, 'benchmark_summary.json'))

    # Print summary in a table
    table = tabulate(list(metrics_result.items()), headers=['Key', 'Value'], tablefmt='grid')
    logger.info('\nBenchmarking summary:\n' + table)

    # Get percentile results
    percentile_result = get_percentile_results(result_db_path, args.percentile_sort_metric)
    if percentile_result:
        write_json_file(transpose_results(percentile_result), os.path.join(result_path, 'benchmark_percentile.json'))
        # Print percentile results in a table
        table = tabulate(percentile_result, headers='keys', tablefmt='pretty')
        logger.info('\nPercentile results:\n' + table)

    if args.dataset.startswith('speed_benchmark'):
        speed_benchmark_result(result_db_path)

    logger.info(f'Save the summary to: {result_path}')

    return metrics_result, percentile_result


def speed_benchmark_result(result_db_path: str):
    query_sql = f"""
        SELECT
            {DatabaseColumns.PROMPT_TOKENS},
            ROUND(AVG({DatabaseColumns.COMPLETION_TOKENS} / {DatabaseColumns.LATENCY}), 2) AS avg_completion_token_per_second,
            ROUND(AVG({DatabaseColumns.MAX_GPU_MEMORY_COST}), 2)
        FROM
            result
        WHERE
            {DatabaseColumns.SUCCESS} = 1 AND {DatabaseColumns.LATENCY} > 0
        GROUP BY
            {DatabaseColumns.PROMPT_TOKENS}
    """  # noqa: E501

    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        cursor.execute(query_sql)
        rows = cursor.fetchall()

    # Prepare data for tabulation
    headers = ['Prompt Tokens', 'Speed(tokens/s)', 'GPU Memory(GB)']
    data = [dict(zip(headers, row)) for row in rows]

    # Print results in a table
    table = tabulate(data, headers='keys', tablefmt='pretty')
    logger.info('\nSpeed Benchmark Results:\n' + table)

    # Write results to JSON file
    result_path = os.path.dirname(result_db_path)
    write_json_file(data, os.path.join(result_path, 'speed_benchmark.json'))
