#!/usr/bin/env python3
"""Compare reconstructed/saved numeric evidence with the pinned paper references.

This verifies reported values, not full model retraining or bootstrap resampling.
"""
import argparse
import ast
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CI_PATTERN = r'\\resultci\{([^{}]+)\}\{([^{}]+)\}\{([^{}]+)\}'


def normalized_number(text):
    return text.replace('\\tableminus', '-').replace('$', '').replace('\\,', '').replace(' ', '').replace('\\%', '')


def check_rounded(actual, expected, decimals):
    """Respect printed precision; reject nonfinite values and larger deviations."""
    actual, expected = float(actual), float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError('Nonfinite paper comparison')
    if abs(actual - expected) > .500001 * 10 ** -decimals:
        raise ValueError(f'{actual} differs from printed {expected} at {decimals} decimals')


def check_cells(actual, expected):
    if len(actual) != len(expected):
        raise ValueError(f'CI cell counts differ: {len(actual)} != {len(expected)}')
    for i, (left, right) in enumerate(zip(actual, expected)):
        if len(left) != 3 or len(right) != 3:
            raise ValueError(f'Malformed CI cell {i}')
        for value, printed in zip(left, right):
            text = normalized_number(str(printed))
            decimals = len(text.split('.')[1]) if '.' in text else 0
            try:
                check_rounded(value, text, decimals)
            except ValueError as error:
                raise ValueError(f'CI cell {i}: {error}') from error


def verify(reference, aggregates):
    errors, covered, checks = [], set(), 0

    def location(source):
        prefix, relative = source.split('/', 1)
        return (aggregates if prefix == 'reproduced' else ROOT / 'results/evidence') / relative

    @lru_cache(maxsize=None)
    def csv_rows(source):
        with location(source).open() as stream:
            return list(csv.DictReader(stream))

    @lru_cache(maxsize=None)
    def json_value(source):
        return json.loads(location(source).read_text())

    for target in reference['quantities']:
        try:
            rows = csv_rows(target['source']) if target['kind'] == 'csv' else json_value(target['source'])
            for key in target['path']:
                rows = rows[key]
            selected = [r for r in rows if all(r.get(k) == v for k, v in target['where'].items())]
            if not selected or (target['expected_rows'] is not None and len(selected) != target['expected_rows']):
                raise ValueError('Missing/duplicate selected rows: ' + str(len(selected)))
            values = [float(r[target['metric']]) for r in selected]
            mode = target['aggregate']
            if mode == 'single':
                if len(values) != 1:
                    raise ValueError('Expected exactly one metric row')
                actual = values[0]
            elif mode == 'mean':
                actual = sum(values) / len(values)
            elif mode == 'sum':
                actual = sum(values)
            elif mode == 'weighted':
                weights = [int(r['n_test']) for r in selected]
                actual = sum(v * w for v, w in zip(values, weights)) / sum(weights)
            else:
                raise ValueError('Unknown aggregation: ' + mode)
            check_rounded(actual * target['scale'], target['expected'], target['decimals'])
            covered.add(target['table'])
            checks += 1
        except (ValueError, KeyError, OSError) as error:
            errors.append({'table': target['table'], 'source': target['source'],
                           'selection': target['where'], 'metric': target['metric'], 'error': str(error)})

    for table in reference['ci_tables']:
        try:
            cells = re.findall(CI_PATTERN, (ROOT / table['source']).read_text())
            actual = [[float(normalized_number(v)) for v in row] for row in cells]
            check_cells(actual, table['expected_ci_cells'])
            covered.update(table['tables'])
            checks += 3 * len(cells)
        except (ValueError, OSError) as error:
            errors.append({'table_file': table['reference_source'], 'error': str(error)})

    family = reference['family_ci']
    try:
        data = json.loads((ROOT / family['source']).read_text())
        actual = []
        for dataset in data['datasets']:
            for row in [*dataset['family_rows'], dataset['query_weighted'], dataset['equal_family']]:
                for method in ['gte', 'l21', 'difference']:
                    actual.append([row[method][key] for key in ['mean', 'low', 'high']])
        check_cells(actual, family['expected_ci_cells'])
        covered.update(family['tables'])
        checks += 3 * len(actual)
    except (ValueError, OSError, KeyError) as error:
        errors.append({'table': 'family-routing', 'error': str(error)})

    for target in reference['mlp_protocol']:
        value = json.loads((ROOT / target['source']).read_text())
        for key in target['path']:
            value = value[key]
        if value != target['expected']:
            errors.append({'table': 'tab:mlp-settings', 'field': target['path'], 'error': 'Configuration differs'})
        else:
            checks += 1
            covered.add('tab:mlp-settings')
    # Compare the price pairs by model identity, preserving the paper row order.
    tree = ast.parse((ROOT / 'src/prepare_data.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sprout')
    assignment = next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'prices' for t in n.targets))
    prices = ast.literal_eval(assignment.value)
    order = ['aws-claude-3-5-sonnet-v1', 'aws-titan-text-premier-v1', 'openai-gpt-4o', 'openai-gpt-4o-mini',
             'wxai-granite-3-2b-instruct-8k-max-tokens', 'wxai-granite-3-8b-instruct-8k-max-tokens',
             'wxai-llama-3-1-8b-instruct', 'wxai-llama-3-1-70b-instruct', 'wxai-llama-3-2-1b-instruct',
             'wxai-llama-3-2-3b-instruct', 'wxai-llama-3-3-70b-instruct', 'wxai-llama-3-405b-instruct',
             'wxai-mixtral-8x7b-instruct-v01']
    if [list(prices[name]) for name in order] != reference['sprout_price_rows']:
        errors.append({'table': 'tab:sprout-prices', 'error': 'Price configuration differs'})
    else:
        checks += 26
        covered.add('tab:sprout-prices')
    missing = sorted(set(reference['table_labels']) - covered)
    if missing:
        errors.append({'error': 'Unverified table coverage', 'tables': missing})
    return {'passed': not errors, 'numeric_checks': checks, 'tables_covered': len(covered),
            'expected_tables': len(reference['table_labels']), 'errors': errors,
            'reference': reference['reference'],
            'scope': 'Reconstructed fold aggregates, supplied CI endpoints and configurations compared with printed paper values. No full model retraining or bootstrap resampling.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregates', type=Path, default=ROOT / 'results/reproduced')
    parser.add_argument('--skip-rebuild', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced/paper-verification.json')
    args = parser.parse_args()
    if not args.skip_rebuild:
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/reproduce_results.py'),
                        '--output', str(args.aggregates)], check=True)
    reference = json.loads((ROOT / 'configs/paper_reference.json').read_text())
    report = verify(reference, args.aggregates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
