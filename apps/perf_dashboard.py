"""Streamlit UI for launching evalscope performance benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

from evalscope.perf.arguments import add_argument
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.utils.db_util import (
    DEFAULT_PERCENTILES,
    PERCENTILE_METRIC_LOOKUP,
    PERCENTILE_METRIC_ORDER,
    PercentileMetrics,
)


HISTORY_DIR = Path.home() / '.evalscope'
HISTORY_PATH = HISTORY_DIR / 'perf_gui_history.json'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    add_argument(parser)
    return parser


PARSER = _build_parser()
ACTION_BY_DEST = {action.dest: action for action in PARSER._actions if action.dest != 'help'}

try:  # pragma: no cover - compatibility shim for Python < 3.9
    from argparse import BooleanOptionalAction  # type: ignore
except ImportError:  # pragma: no cover
    class BooleanOptionalAction(argparse.Action):  # type: ignore
        pass


TAB_CONFIG: List[Tuple[str, Iterable[str]]] = [
    (
        'Core',
        (
            'model',
            'api',
            'url',
            'api_key',
            'dataset',
            'dataset_path',
            'prompt',
            'number',
            'parallel',
            'rate',
            'outputs_dir',
            'percentile_sort_metric',
            'seed',
            'max_tokens',
            'temperature',
            'stream',
        ),
    ),
    (
        'Prompt & Dataset',
        (
            'max_prompt_length',
            'min_prompt_length',
            'prefix_length',
            'query_template',
            'apply_chat_template',
            'stop',
            'stop_token_ids',
        ),
    ),
    (
        'Response',
        (
            'frequency_penalty',
            'repetition_penalty',
            'logprobs',
            'n_choices',
            'min_tokens',
            'top_p',
            'top_k',
            'extra_args',
        ),
    ),
    (
        'Connection',
        (
            'port',
            'headers',
            'connect_timeout',
            'read_timeout',
            'no_test_connection',
        ),
    ),
    (
        'Logging',
        (
            'log_every_n_query',
            'debug',
            'wandb_api_key',
            'swanlab_api_key',
            'name',
        ),
    ),
    (
        'Vision',
        (
            'image_width',
            'image_height',
            'image_format',
            'image_num',
            'image_patch_size',
        ),
    ),
    (
        'Advanced',
        (
            'attn_implementation',
            'tokenizer_path',
            'sleep_interval',
        ),
    ),
]


def _remaining_destinations() -> List[str]:
    defined = {dest for _, dests in TAB_CONFIG for dest in dests}
    return [dest for dest in ACTION_BY_DEST if dest not in defined]


for leftover in _remaining_destinations():
    # ensure every argument is surfaced somewhere
    for name, dests in TAB_CONFIG:
        if name == 'Advanced':
            TAB_CONFIG[TAB_CONFIG.index((name, dests))] = (name, tuple(list(dests) + [leftover]))
            break


def _load_history() -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        with HISTORY_PATH.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _persist_history(history: List[Dict[str, Any]]) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open('w', encoding='utf-8') as fh:
        json.dump(history[-20:], fh, indent=2, ensure_ascii=False)


def _initial_value(action, overrides: Dict[str, Any]) -> Any:
    value = overrides.get(action.dest, st.session_state.get(f'arg_{action.dest}'))
    if value is None:
        value = action.default
    if isinstance(value, list):
        return ', '.join(str(item) for item in value)
    if isinstance(value, dict):
        return '\n'.join(f'{k}={v}' for k, v in value.items())
    if isinstance(value, bool):
        return value
    if value is None:
        return ''
    return value


def _render_text_input(container, action, *, multiline: bool = False, overrides: Dict[str, Any]):
    key = f'arg_{action.dest}'
    default_value = _initial_value(action, overrides)
    help_text = action.help or ''
    if multiline:
        container.text_area(action.dest, value=default_value, key=key, help=help_text)
    else:
        container.text_input(action.dest, value=str(default_value), key=key, help=help_text)


def _render_bool_checkbox(container, action, overrides: Dict[str, Any]):
    key = f'arg_{action.dest}'
    default_value = overrides.get(action.dest)
    if default_value is None:
        default_value = bool(action.default)
    container.checkbox(action.dest, value=bool(default_value), key=key, help=action.help or '')


def _render_boolean_optional(container, action, overrides: Dict[str, Any]):
    key = f'arg_{action.dest}'
    select_key = f'{key}_select'
    default_value = overrides.get(action.dest, action.default)
    options = {
        f'Default ({action.default})' if action.default is not None else 'Default': None,
        'Enable': True,
        'Disable': False,
    }
    reverse_lookup = {v: label for label, v in options.items()}
    initial_label = reverse_lookup.get(
        default_value,
        'Default' if action.default is None else f'Default ({action.default})'
    )
    selection = container.selectbox(
        action.dest,
        list(options.keys()),
        index=list(options.keys()).index(initial_label),
        help=action.help or '',
        key=select_key
    )
    st.session_state[key] = options[selection]


def _render_field(container, action, overrides: Dict[str, Any]):
    action_type = type(action).__name__
    if action.dest == 'percentile_sort_metric':
        key = f'arg_{action.dest}'
        available = [''] + sorted(set(PERCENTILE_METRIC_LOOKUP.values()))
        default_value = st.session_state.get(key, action.default or '')
        try:
            default_index = available.index(default_value or '')
        except ValueError:
            default_index = 0
        selection = container.selectbox(
            'Percentile sort metric',
            available,
            index=default_index,
            help=action.help or ''
        )
        st.session_state[key] = selection or ''
    elif isinstance(action, BooleanOptionalAction):
        _render_boolean_optional(container, action, overrides)
    elif action_type in {'_StoreTrueAction', '_StoreFalseAction'}:
        _render_bool_checkbox(container, action, overrides)
    elif action.dest in {'headers', 'extra_args'}:
        _render_text_input(container, action, multiline=True, overrides=overrides)
    else:
        _render_text_input(container, action, overrides=overrides)


def _cast_list(value: str, *, value_type) -> Optional[List[Any]]:
    if not value:
        return None
    items = [item.strip() for item in value.replace('\n', ',').split(',') if item.strip()]
    if not items:
        return None
    try:
        return [value_type(item) for item in items]
    except ValueError:
        raise ValueError(f'Unable to parse list items from "{value}"')


def _parse_headers(value: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not value:
        return result
    for line in value.splitlines():
        if not line.strip():
            continue
        if '=' not in line:
            raise ValueError(f'Header line "{line}" is missing "="')
        key, val = line.split('=', 1)
        result[key.strip()] = val.strip()
    return result


def _gather_arguments() -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    errors: List[str] = []

    for dest, action in ACTION_BY_DEST.items():
        key = f'arg_{dest}'
        if key not in st.session_state:
            continue
        raw_value = st.session_state[key]

        if isinstance(raw_value, str):
            raw_value = raw_value.strip()

        if dest in {'headers'}:
            try:
                parsed = _parse_headers(raw_value)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if parsed:
                args[dest] = parsed
            continue

        if dest in {'extra_args'}:
            if raw_value:
                try:
                    args[dest] = json.loads(raw_value)
                except json.JSONDecodeError as exc:
                    errors.append(f'extra_args JSON error: {exc}')
            continue

        nargs = getattr(action, 'nargs', None)
        expected_type = getattr(action, 'type', str)

        if isinstance(raw_value, str) and not raw_value:
            continue

        if type(action).__name__ in {'_StoreTrueAction', '_StoreFalseAction'}:
            args[dest] = bool(raw_value)
            continue

        if isinstance(action, BooleanOptionalAction):
            if raw_value is None:
                continue
            args[dest] = bool(raw_value)
            continue

        if nargs in ('+', '*') or isinstance(action.default, list):
            try:
                list_value = _cast_list(raw_value if isinstance(raw_value, str) else ','.join(raw_value), value_type=expected_type or str)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if list_value is not None:
                args[dest] = list_value
            continue

        if expected_type is None:
            expected_type = str

        if raw_value is None:
            continue

        if expected_type in (int, float, str):
            try:
                args[dest] = expected_type(raw_value)
            except (TypeError, ValueError):
                errors.append(f'Invalid value for {dest}: {raw_value}')
        else:
            args[dest] = raw_value

    missing_required = [dest for dest, action in ACTION_BY_DEST.items() if getattr(action, 'required', False) and dest not in args]
    if missing_required:
        errors.append(f'Missing required fields: {", ".join(missing_required)}')

    if errors:
        raise ValueError('\n'.join(errors))

    if args.get('percentile_sort_metric') == '':
        args.pop('percentile_sort_metric')

    return args


def _format_history_label(entry: Dict[str, Any]) -> str:
    args = entry.get('args', {})
    name = args.get('name') or args.get('model') or 'run'
    ts = entry.get('timestamp', '')
    return f"{ts} • {name}"


def _save_run(args: Dict[str, Any], outputs_dir: Optional[Path]):
    history = st.session_state.setdefault('perf_history', _load_history())
    history.append(
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'args': args,
            'outputs_dir': str(outputs_dir) if outputs_dir else None,
        }
    )
    st.session_state['perf_history'] = history
    _persist_history(history)


def _set_form_state(args: Dict[str, Any]):
    for dest, action in ACTION_BY_DEST.items():
        key = f'arg_{dest}'
        value = args.get(dest, action.default)
        if isinstance(value, list):
            value = ', '.join(str(item) for item in value)
        elif isinstance(value, dict):
            value = '\n'.join(f'{k}={v}' for k, v in value.items())
        elif value is None and type(action).__name__ == '_StoreTrueAction':
            value = False
        st.session_state[key] = value
        if isinstance(action, BooleanOptionalAction):
            options = {
                f'Default ({action.default})' if action.default is not None else 'Default': None,
                'Enable': True,
                'Disable': False,
            }
            reverse_lookup = {v: label for label, v in options.items()}
            label = reverse_lookup.get(value, list(options.keys())[0])
            st.session_state[f'{key}_select'] = label


def _latest_output_path(base_dir: str | None) -> Optional[Path]:
    if not base_dir:
        return None
    path = Path(base_dir)
    if not path.exists():
        return None
    candidates = sorted(path.rglob('benchmark_summary.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0].parent


def _display_metrics(metrics: Dict[str, Any]):
    st.markdown('#### Summary Metrics')
    items = list(metrics.items())
    for idx in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (label, value) in zip(cols, items[idx:idx + 3]):
            col.metric(label, value)


def _display_percentiles(percentiles: Dict[str, List[Any]]):
    if not percentiles:
        return
    st.markdown('#### Percentile Metrics')
    df = pd.DataFrame(percentiles)
    st.dataframe(df, use_container_width=True, hide_index=True)

    chart_df = df.set_index(PercentileMetrics.PERCENTILES)
    chart_df = chart_df[[col for col in PERCENTILE_METRIC_ORDER if col in chart_df.columns]]
    st.line_chart(chart_df, height=280)


def _display_results(results):
    if isinstance(results, tuple):
        metrics, percentiles = results
        _display_metrics(metrics)
        _display_percentiles(percentiles)
    elif isinstance(results, list):
        for idx, (metrics, percentiles) in enumerate(results, start=1):
            st.markdown(f'### Run {idx}')
            _display_metrics(metrics)
            _display_percentiles(percentiles)


def main():
    st.set_page_config(page_title='Evalscope Perf Dashboard', layout='wide')

    st.markdown(
        """
        <style>
            body {background-color:#f5f5f7;}
            [data-testid="stAppViewContainer"] {background-color:#f5f5f7;}
            [data-testid="stSidebar"] {background-color:#ffffff; border-right:1px solid rgba(0,0,0,0.05);}
            h1, h2, h3, h4 {font-weight:600; color:#1c1c1e;}
            .stButton>button {background:#0a84ff; color:white; border-radius:20px; border:none; padding:0.6rem 1.6rem; font-weight:600;}
            .stButton>button:hover {background:#006edc;}
            .stSelectbox label, .stTextInput label, .stNumberInput label, .stCheckbox label {font-weight:500; color:#3a3a3c;}
            .block-container {padding-top:2rem; padding-bottom:4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Evalscope Performance Dashboard")
    st.markdown(
        "<p style='color:#6e6e73;'>Configure and launch load tests with a calm, minimal interface.</p>",
        unsafe_allow_html=True,
    )

    history = st.session_state.setdefault('perf_history', _load_history())

    with st.sidebar:
        st.markdown('#### Recent Runs')
        options = ['Create new'] + [_format_history_label(item) for item in history]
        selected = st.selectbox('', options, label_visibility='collapsed')
        if selected != 'Create new':
            idx = options.index(selected) - 1
            if st.button('Load configuration', use_container_width=True):
                _set_form_state(history[idx]['args'])
                st.experimental_rerun()

        st.markdown('---')
        st.caption('Percentile metrics enabled:')
        st.caption(', '.join(sorted(set(PERCENTILE_METRIC_LOOKUP.values()))))

    overrides: Dict[str, Any] = {}

    with st.form('perf_form'):
        primary_group, *other_groups = TAB_CONFIG

        st.markdown('#### Core Settings')
        core_columns = st.columns(3)
        for idx, dest in enumerate(primary_group[1]):
            container = core_columns[idx % 3]
            action = ACTION_BY_DEST.get(dest)
            if not action:
                continue
            _render_field(container, action, overrides)

        for label, dests in other_groups:
            with st.expander(label, expanded=False):
                cols = st.columns(2)
                for idx, dest in enumerate(dests):
                    action = ACTION_BY_DEST.get(dest)
                    if not action:
                        continue
                    container = cols[idx % 2]
                    _render_field(container, action, overrides)

        submitted = st.form_submit_button('Run Benchmark', type='primary')

    if submitted:
        try:
            args_dict = _gather_arguments()
        except ValueError as exc:
            st.error(str(exc))
            return

        st.write('Launching benchmark...')
        try:
            with st.spinner('Running benchmark, please wait...'):
                results = run_perf_benchmark(args_dict)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            st.error(f'Benchmark failed: {exc}')
            st.exception(exc)
            return

        st.success('Benchmark completed')
        _display_results(results)

        outputs_dir = args_dict.get('outputs_dir')
        latest_path = _latest_output_path(outputs_dir)
        if latest_path:
            st.info(f'Results saved under: `{latest_path}`')

        _save_run(args_dict, latest_path)


if __name__ == '__main__':
    main()
