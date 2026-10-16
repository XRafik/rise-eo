"""Characterisation pins for the P14c split (#145 C13, A19), written against the
unchanged code and kept unedited across the split.

- `bundle._check_no_unembedded_siblings(root, rels)`: the full `str(exc)` (or "OK" for no
  raise) for each case below.
- `download_cli.main(argv)`: return code, stdout, stderr and `_result.json`'s full text for
  each case below, with `cdse` stubbed (no network) and a fake clock.

The expected values at the bottom were computed once from the unchanged code.
"""

import contextlib
import io
import json
import os

import pytest

from rise_eo.model import bundle
from rise_eo.sources import cdse, download_cli

# --- bundle: _check_no_unembedded_siblings -------------------------------------------------

# name -> (files under <tmp>/root, rels passed to the check)
BUNDLE_CASES = {
    "direct_sibling": ({"main.py": "import zqsib\n", "zqsib.py": ""}, ["main.py"]),
    "sibling_package": (
        {"main.py": "import zqpkg\n", "zqpkg/__init__.py": ""}, ["main.py"],
    ),
    "file_beats_package_unembedded": (
        {"main.py": "import zqboth\n", "zqboth.py": "", "zqboth/__init__.py": ""},
        ["main.py"],
    ),
    "file_beats_package_embedded": (
        {"main.py": "import zqboth\n", "zqboth.py": "", "zqboth/__init__.py": ""},
        ["main.py", "zqboth.py"],
    ),
    "transitive": (
        {"zqa.py": "import zqb\n", "zqb.py": "import zqc\n", "zqc.py": ""},
        ["zqa.py", "zqb.py"],
    ),
    "transitive_all_embedded": (
        {"zqa.py": "import zqb\n", "zqb.py": "import zqc\n", "zqc.py": ""},
        ["zqa.py", "zqb.py", "zqc.py"],
    ),
    "lifo_two_missing": (
        {"m1.py": "import zqone\n", "m2.py": "import zqtwo\n", "zqone.py": "", "zqtwo.py": ""},
        ["m1.py", "m2.py"],
    ),
    "relative_from_dot": (
        {"main.py": "from . import zqrel\n", "zqrel.py": ""}, ["main.py"],
    ),
    "relative_from_dot_name": (
        {"main.py": "from .zqrel import y\n", "zqrel.py": ""}, ["main.py"],
    ),
    "import_fsd": ({"main.py": "import rise_eo\n", "fsd.py": ""}, ["main.py"]),
    "stdlib_shadow": ({"main.py": "import json\n", "json.py": ""}, ["main.py"]),
    "installed_shadow": ({"main.py": "import numpy\n", "numpy.py": ""}, ["main.py"]),
    "resolves_nowhere": ({"main.py": "import zqnowhere\n"}, ["main.py"]),
    "syntax_error": (
        {"bad.py": "def (:\n", "zqsib.py": ""}, ["bad.py"],
    ),
    "syntax_error_then_good": (
        {"bad.py": "def (:\n", "main.py": "import zqsib\n", "zqsib.py": ""},
        ["bad.py", "main.py"],
    ),
    "non_py_entry": (
        {"notes.txt": "import zqsib\n", "zqsib.py": ""}, ["notes.txt"],
    ),
    "import_in_function_body": (
        {"main.py": "def f():\n    import zqsib\n", "zqsib.py": ""}, ["main.py"],
    ),
    "import_dotted": (
        {"main.py": "import zqdot.sub\n", "zqdot.py": ""}, ["main.py"],
    ),
    "from_dotted": (
        {"main.py": "from zqdot.sub import c\n", "zqdot/__init__.py": ""}, ["main.py"],
    ),
    "import_multi_alias": (
        {"main.py": "import json, zqsib as s\n", "zqsib.py": ""}, ["main.py"],
    ),
}

# name -> (files, rels): the list of names `_is_installed_distribution` is called with
BUNDLE_SPY_CASES = {
    "name_checked_once": (
        {
            "f1.py": "import zqx\nimport json\n",
            "f2.py": "import zqx\nimport zqy\n",
            "zqx.py": "",
            "zqy.py": "",
        },
        ["f1.py", "f2.py", "zqx.py", "zqy.py"],
    ),
    "name_checked_once_reordered": (
        {
            "f1.py": "import zqx\nimport json\n",
            "f2.py": "import zqx\nimport zqy\n",
            "zqx.py": "",
            "zqy.py": "",
        },
        ["zqx.py", "f2.py", "zqy.py", "f1.py"],
    ),
}


def _write_tree(base, files):
    for rel, text in files.items():
        path = os.path.join(base, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def _observe_bundle(case, base):
    files, rels = BUNDLE_CASES[case]
    base = os.path.realpath(base)
    root = os.path.join(base, "root")
    _write_tree(root, files)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(base)
        try:
            bundle._check_no_unembedded_siblings(root, rels)
        except ValueError as exc:
            return str(exc)
    return "OK"


def _observe_bundle_spy(case, base):
    files, rels = BUNDLE_SPY_CASES[case]
    base = os.path.realpath(base)
    root = os.path.join(base, "root")
    _write_tree(root, files)
    seen = []
    real = bundle._is_installed_distribution

    def spy(name):
        seen.append(name)
        return real(name)

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(base)
        mp.setattr(bundle, "_is_installed_distribution", spy)
        try:
            bundle._check_no_unembedded_siblings(root, rels)
            outcome = "OK"
        except ValueError as exc:
            outcome = str(exc)
    return {"outcome": outcome, "called_with": seen}


# --- download_cli: main(argv) --------------------------------------------------------------

_PLAN = {
    "needed_count": 7, "present_count": 2, "missing_count": 5,
    "missing_ids": ["a"],
    "download_params": {
        "roi": "roi.geojson", "startdate": "2018-04-01", "enddate": "2018-06-01",
        "bands": ["B04"], "max_tiles": 7, "max_cloudcover": None,
        "dst_folderpath": "dst",
    },
}


def _dr(**kw):
    kw.setdefault("successful_count", 5)
    kw.setdefault("total_count", 5)
    return cdse.DownloadResult(**kw)


_BIG = dict(
    bytes_downloaded=2_345_000_000, transfer_seconds=40.0, transfer_wall_seconds=10.0,
    convert_seconds=3.3, skipped_count=1,
)

# name -> dict(extra=<argv extras>, results=[...], plan_raises, resume_raises, expected=<json>)
CLI_CASES = {
    "dry_run": dict(extra=["--dry-run"]),
    "dry_run_expected_json": dict(
        extra=["--dry-run"], expected={"failed": 3, "note": "dry"},
    ),
    "ok": dict(results=[_dr(**_BIG)]),
    "ok_quiet_noprobe": dict(extra=["--quiet", "--no-probe"], results=[_dr(**_BIG)]),
    "ok_recovered_transient": dict(results=[
        _dr(failed_count=2, successful_count=3, **_BIG),
        _dr(bytes_downloaded=100_000_000, transfer_seconds=2.0, transfer_wall_seconds=1.0),
    ]),
    "stopped": dict(results=[_dr(stopped=True, successful_count=2, **_BIG)]),
    "empty_results": dict(results=[]),
    "empty_results_stop_file_exists": dict(extra=["--stop-file", "<tmp>/stop"], results=[],
                                           touch_stop=True),
    "failed_terminal_count": dict(results=[_dr(failed_count=3, successful_count=2, **_BIG)]),
    "failed_circuit_tripped": dict(results=[_dr(circuit_tripped=True, **_BIG)]),
    "failed_pool_broken_with_failures": dict(
        results=[_dr(pool_broken=True, failed_count=1, **_BIG)],
    ),
    "failed_all_three": dict(results=[
        _dr(failed_count=4, circuit_tripped=True, pool_broken=True, **_BIG),
    ]),
    "pool_broken_without_failures_is_ok": dict(results=[_dr(pool_broken=True, **_BIG)]),
    "failed_and_stopped_is_stopped": dict(
        results=[_dr(stopped=True, failed_count=2, **_BIG)],
    ),
    "zero_seconds": dict(results=[_dr(bytes_downloaded=5_000_000)]),
    "real_expected_json": dict(
        results=[_dr(**_BIG)], expected={"failed": 99, "custom": "x"},
    ),
    "crash_real": dict(resume_raises=RuntimeError("boom")),
    "crash_dry_run": dict(
        extra=["--dry-run"], plan_raises=ValueError("nope"), expected={"k": 1},
    ),
}


class _Clock:
    def __init__(self):
        self.n = 0

    def time(self):
        self.n += 1
        return 100.0 if self.n == 1 else 112.34


def _observe_cli(case, base):
    spec = CLI_CASES[case]
    out, err = io.StringIO(), io.StringIO()
    result_json = os.path.join(base, "_result.json")
    argv = [
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", base, "--catalog", os.path.join(base, "c.parquet"),
        "--max-tiles", "7", "--result-json", result_json,
        "--creds", os.path.join(base, "creds.json"),
    ] + [a.replace("<tmp>", base) for a in spec.get("extra", [])]
    if "expected" in spec:
        exp_path = os.path.join(base, "expected.json")
        with open(exp_path, "w") as f:
            json.dump(spec["expected"], f)
        argv += ["--expected-json", exp_path]
    if spec.get("touch_stop"):
        open(os.path.join(base, "stop"), "w").close()

    def plan(*a, **k):
        if "plan_raises" in spec:
            raise spec["plan_raises"]
        return _PLAN

    def resume(*a, **k):
        if "resume_raises" in spec:
            raise spec["resume_raises"]
        return spec["results"]

    dummy = dict(sh_client_id="i", sh_client_secret="s", s3_access_key="a", s3_secret_key="k")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(base)
        mp.setattr(download_cli, "time", _Clock())
        mp.setattr(download_cli.CdseCredentials, "from_json",
                   classmethod(lambda cls, fp, **kw: cls(**dummy)))
        mp.setattr(download_cli.cdse, "plan_download", plan)
        mp.setattr(download_cli.cdse, "probe_throughput", lambda *a, **k: (12.34, 1000, 0.1))
        mp.setattr(download_cli.cdse, "download_resume", resume)
        rc, raised = None, None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = download_cli.main(argv)
            except Exception as exc:  # noqa: BLE001
                raised = f"{type(exc).__name__}: {exc}"
    with open(result_json) as f:
        text = f.read()
    norm = lambda s: s.replace(base, "<tmp>")  # noqa: E731
    return {
        "rc": rc, "raised": raised,
        "stdout": norm(out.getvalue()), "stderr": norm(err.getvalue()),
        "result_text": norm(text),
    }


# --- the pins ------------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(BUNDLE_CASES))
def test_bundle_sibling_check_pin(case, tmp_path):
    assert _observe_bundle(case, str(tmp_path)) == EXPECTED_BUNDLE[case]


@pytest.mark.parametrize("case", sorted(BUNDLE_SPY_CASES))
def test_bundle_sibling_check_names_pin(case, tmp_path):
    assert _observe_bundle_spy(case, str(tmp_path)) == EXPECTED_BUNDLE_SPY[case]


@pytest.mark.parametrize("case", sorted(CLI_CASES))
def test_download_cli_pin(case, tmp_path):
    assert _observe_cli(case, str(tmp_path)) == EXPECTED_CLI[case]


# --- expected values, computed once from the unchanged code --------------------------------

EXPECTED_BUNDLE = {'direct_sibling': "main.py imports 'zqsib', which resolves to './root/zqsib.py' but is not "
                   'embedded. The node would raise ModuleNotFoundError after a cold start.\n'
                   "Fix: code=['./root/main.py', './root/zqsib.py']",
 'file_beats_package_embedded': 'OK',
 'file_beats_package_unembedded': "main.py imports 'zqboth', which resolves to "
                                  "'./root/zqboth.py' but is not embedded. The node would "
                                  'raise ModuleNotFoundError after a cold start.\n'
                                  "Fix: code=['./root/main.py', './root/zqboth.py']",
 'from_dotted': "main.py imports 'zqdot', which resolves to './root/zqdot/__init__.py' but is "
                'not embedded. The node would raise ModuleNotFoundError after a cold start.\n'
                "Fix: code=['./root/main.py', './root/zqdot/__init__.py']",
 'import_dotted': "main.py imports 'zqdot', which resolves to './root/zqdot.py' but is not "
                  'embedded. The node would raise ModuleNotFoundError after a cold start.\n'
                  "Fix: code=['./root/main.py', './root/zqdot.py']",
 'import_fsd': 'OK',
 'import_in_function_body': "main.py imports 'zqsib', which resolves to './root/zqsib.py' but "
                            'is not embedded. The node would raise ModuleNotFoundError after '
                            'a cold start.\n'
                            "Fix: code=['./root/main.py', './root/zqsib.py']",
 'import_multi_alias': "main.py imports 'zqsib', which resolves to './root/zqsib.py' but is "
                       'not embedded. The node would raise ModuleNotFoundError after a cold '
                       'start.\n'
                       "Fix: code=['./root/main.py', './root/zqsib.py']",
 'installed_shadow': 'OK',
 'lifo_two_missing': "m2.py imports 'zqtwo', which resolves to './root/zqtwo.py' but is not "
                     'embedded. The node would raise ModuleNotFoundError after a cold start.\n'
                     "Fix: code=['./root/m1.py', './root/m2.py', './root/zqtwo.py']",
 'non_py_entry': 'OK',
 'relative_from_dot': 'OK',
 'relative_from_dot_name': 'OK',
 'resolves_nowhere': 'OK',
 'sibling_package': "main.py imports 'zqpkg', which resolves to './root/zqpkg/__init__.py' "
                    'but is not embedded. The node would raise ModuleNotFoundError after a '
                    'cold start.\n'
                    "Fix: code=['./root/main.py', './root/zqpkg/__init__.py']",
 'stdlib_shadow': 'OK',
 'syntax_error': 'OK',
 'syntax_error_then_good': "main.py imports 'zqsib', which resolves to './root/zqsib.py' but "
                           'is not embedded. The node would raise ModuleNotFoundError after a '
                           'cold start.\n'
                           "Fix: code=['./root/bad.py', './root/main.py', './root/zqsib.py']",
 'transitive': "zqb.py imports 'zqc', which resolves to './root/zqc.py' but is not embedded. "
               'The node would raise ModuleNotFoundError after a cold start.\n'
               "Fix: code=['./root/zqa.py', './root/zqb.py', './root/zqc.py']",
 'transitive_all_embedded': 'OK'}

EXPECTED_BUNDLE_SPY = {'name_checked_once': {'called_with': ['zqx', 'zqy', 'json'], 'outcome': 'OK'},
 'name_checked_once_reordered': {'called_with': ['zqx', 'json', 'zqy'], 'outcome': 'OK'}}

EXPECTED_CLI = {'crash_dry_run': {'raised': 'ValueError: nope',
                   'rc': None,
                   'result_text': '{\n'
                                  '  "step": "download-confirm-run",\n'
                                  '  "status": "failed",\n'
                                  '  "pass": 0,\n'
                                  '  "metrics": {},\n'
                                  '  "expected": {\n'
                                  '    "k": 1\n'
                                  '  },\n'
                                  '  "error": "ValueError(\'nope\')"\n'
                                  '}',
                   'stderr': '',
                   'stdout': ''},
 'crash_real': {'raised': 'RuntimeError: boom',
                'rc': None,
                'result_text': '{\n'
                               '  "step": "download-confirm-run",\n'
                               '  "status": "failed",\n'
                               '  "pass": 0,\n'
                               '  "metrics": {},\n'
                               '  "expected": {\n'
                               '    "failed": 0,\n'
                               '    "stopped": false,\n'
                               '    "circuit_tripped": false,\n'
                               '    "pool_broken": false\n'
                               '  },\n'
                               '  "error": "RuntimeError(\'boom\')"\n'
                               '}',
                'stderr': '',
                'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band file)…\n'
                          '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                          '[rise_eo.download_cli] discovering + planning download…\n'},
 'dry_run': {'raised': None,
             'rc': 0,
             'result_text': '{\n'
                            '  "step": "download-confirm-run",\n'
                            '  "status": "dry-run",\n'
                            '  "pass": 0,\n'
                            '  "metrics": {\n'
                            '    "needed": 7,\n'
                            '    "present": 2,\n'
                            '    "missing": 5\n'
                            '  },\n'
                            '  "expected": {},\n'
                            '  "error": null\n'
                            '}',
             'stderr': '',
             'stdout': 'imagery for this run is not (fully) present in the catalog.\n'
                       '  needed: 7 granules | present: 2 | missing: 5\n'
                       '  run rise_eo.download(\n'
                       '      roi=\'roi.geojson\', startdate="2018-04-01", '
                       'enddate="2018-06-01",\n'
                       '      bands=["B04"], max_tiles=7, max_cloudcover=None, '
                       "dst_folderpath='dst')\n"},
 'dry_run_expected_json': {'raised': None,
                           'rc': 0,
                           'result_text': '{\n'
                                          '  "step": "download-confirm-run",\n'
                                          '  "status": "dry-run",\n'
                                          '  "pass": 0,\n'
                                          '  "metrics": {\n'
                                          '    "needed": 7,\n'
                                          '    "present": 2,\n'
                                          '    "missing": 5\n'
                                          '  },\n'
                                          '  "expected": {\n'
                                          '    "failed": 3,\n'
                                          '    "note": "dry"\n'
                                          '  },\n'
                                          '  "error": null\n'
                                          '}',
                           'stderr': '',
                           'stdout': 'imagery for this run is not (fully) present in the '
                                     'catalog.\n'
                                     '  needed: 7 granules | present: 2 | missing: 5\n'
                                     '  run rise_eo.download(\n'
                                     '      roi=\'roi.geojson\', startdate="2018-04-01", '
                                     'enddate="2018-06-01",\n'
                                     '      bands=["B04"], max_tiles=7, max_cloudcover=None, '
                                     "dst_folderpath='dst')\n"},
 'empty_results': {'raised': None,
                   'rc': 0,
                   'result_text': '{\n'
                                  '  "step": "download-confirm-run",\n'
                                  '  "status": "stopped",\n'
                                  '  "pass": 0,\n'
                                  '  "metrics": {\n'
                                  '    "successful": 0,\n'
                                  '    "failed": 0,\n'
                                  '    "failed_total": 0,\n'
                                  '    "skipped": 0,\n'
                                  '    "gb": 0.0,\n'
                                  '    "transfer_s": 0.0,\n'
                                  '    "transfer_wall_s": 0.0,\n'
                                  '    "convert_s": 0.0,\n'
                                  '    "probe_mb_per_s": 12.34,\n'
                                  '    "aggregate_mb_per_s": 0.0,\n'
                                  '    "wall_transfer_mb_per_s": 0.0,\n'
                                  '    "elapsed_s": 12.3,\n'
                                  '    "stopped": true,\n'
                                  '    "circuit_tripped": false,\n'
                                  '    "pool_broken": false\n'
                                  '  },\n'
                                  '  "expected": {\n'
                                  '    "failed": 0,\n'
                                  '    "stopped": false,\n'
                                  '    "circuit_tripped": false,\n'
                                  '    "pool_broken": false\n'
                                  '  },\n'
                                  '  "error": null\n'
                                  '}',
                   'stderr': '',
                   'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band file)…\n'
                             '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                             '[rise_eo.download_cli] discovering + planning download…\n'
                             '\n'
                             '[rise_eo.download_cli] stopped: successful=0 failed=0 (total across '
                             'passes 0) skipped=0 (0.00 GB) | transfer=0.0s convert=0.0s | '
                             'probe=12.3 MB/s per-stream=0.0 wall=0.0 MB/s | stopped=True '
                             'circuit_tripped=False pool_broken=False\n'},
 'empty_results_stop_file_exists': {'raised': None,
                                    'rc': 0,
                                    'result_text': '{\n'
                                                   '  "step": "download-confirm-run",\n'
                                                   '  "status": "stopped",\n'
                                                   '  "pass": 0,\n'
                                                   '  "metrics": {\n'
                                                   '    "successful": 0,\n'
                                                   '    "failed": 0,\n'
                                                   '    "failed_total": 0,\n'
                                                   '    "skipped": 0,\n'
                                                   '    "gb": 0.0,\n'
                                                   '    "transfer_s": 0.0,\n'
                                                   '    "transfer_wall_s": 0.0,\n'
                                                   '    "convert_s": 0.0,\n'
                                                   '    "probe_mb_per_s": 12.34,\n'
                                                   '    "aggregate_mb_per_s": 0.0,\n'
                                                   '    "wall_transfer_mb_per_s": 0.0,\n'
                                                   '    "elapsed_s": 12.3,\n'
                                                   '    "stopped": true,\n'
                                                   '    "circuit_tripped": false,\n'
                                                   '    "pool_broken": false\n'
                                                   '  },\n'
                                                   '  "expected": {\n'
                                                   '    "failed": 0,\n'
                                                   '    "stopped": false,\n'
                                                   '    "circuit_tripped": false,\n'
                                                   '    "pool_broken": false\n'
                                                   '  },\n'
                                                   '  "error": null\n'
                                                   '}',
                                    'stderr': '[rise_eo.download_cli] warning: stop-file '
                                              '<tmp>/stop already exists — this run will stop '
                                              'immediately without downloading. rm it first '
                                              'to actually download.\n',
                                    'stdout': '[rise_eo.download_cli] probing throughput '
                                              '(downloads 1 band file)…\n'
                                              '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                              '[rise_eo.download_cli] discovering + planning '
                                              'download…\n'
                                              '\n'
                                              '[rise_eo.download_cli] stopped: successful=0 '
                                              'failed=0 (total across passes 0) skipped=0 '
                                              '(0.00 GB) | transfer=0.0s convert=0.0s | '
                                              'probe=12.3 MB/s per-stream=0.0 wall=0.0 MB/s | '
                                              'stopped=True circuit_tripped=False '
                                              'pool_broken=False\n'},
 'failed_all_three': {'raised': None,
                      'rc': 1,
                      'result_text': '{\n'
                                     '  "step": "download-confirm-run",\n'
                                     '  "status": "failed",\n'
                                     '  "pass": 1,\n'
                                     '  "metrics": {\n'
                                     '    "successful": 5,\n'
                                     '    "failed": 4,\n'
                                     '    "failed_total": 4,\n'
                                     '    "skipped": 1,\n'
                                     '    "gb": 2.345,\n'
                                     '    "transfer_s": 40.0,\n'
                                     '    "transfer_wall_s": 10.0,\n'
                                     '    "convert_s": 3.3,\n'
                                     '    "probe_mb_per_s": 12.34,\n'
                                     '    "aggregate_mb_per_s": 58.62,\n'
                                     '    "wall_transfer_mb_per_s": 234.5,\n'
                                     '    "elapsed_s": 12.3,\n'
                                     '    "stopped": false,\n'
                                     '    "circuit_tripped": true,\n'
                                     '    "pool_broken": true\n'
                                     '  },\n'
                                     '  "expected": {\n'
                                     '    "failed": 0,\n'
                                     '    "stopped": false,\n'
                                     '    "circuit_tripped": false,\n'
                                     '    "pool_broken": false\n'
                                     '  },\n'
                                     '  "error": "4 file(s) failed on the terminal pass; '
                                     'circuit breaker tripped; convert pool broken with '
                                     'unresolved failures"\n'
                                     '}',
                      'stderr': '',
                      'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band '
                                'file)…\n'
                                '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                '[rise_eo.download_cli] discovering + planning download…\n'
                                '\n'
                                '[rise_eo.download_cli] failed: successful=5 failed=4 (total '
                                'across passes 4) skipped=1 (2.35 GB) | transfer=40.0s '
                                'convert=3.3s | probe=12.3 MB/s per-stream=58.6 wall=234.5 '
                                'MB/s | stopped=False circuit_tripped=True '
                                'pool_broken=True\n'},
 'failed_and_stopped_is_stopped': {'raised': None,
                                   'rc': 0,
                                   'result_text': '{\n'
                                                  '  "step": "download-confirm-run",\n'
                                                  '  "status": "stopped",\n'
                                                  '  "pass": 1,\n'
                                                  '  "metrics": {\n'
                                                  '    "successful": 5,\n'
                                                  '    "failed": 2,\n'
                                                  '    "failed_total": 2,\n'
                                                  '    "skipped": 1,\n'
                                                  '    "gb": 2.345,\n'
                                                  '    "transfer_s": 40.0,\n'
                                                  '    "transfer_wall_s": 10.0,\n'
                                                  '    "convert_s": 3.3,\n'
                                                  '    "probe_mb_per_s": 12.34,\n'
                                                  '    "aggregate_mb_per_s": 58.62,\n'
                                                  '    "wall_transfer_mb_per_s": 234.5,\n'
                                                  '    "elapsed_s": 12.3,\n'
                                                  '    "stopped": true,\n'
                                                  '    "circuit_tripped": false,\n'
                                                  '    "pool_broken": false\n'
                                                  '  },\n'
                                                  '  "expected": {\n'
                                                  '    "failed": 0,\n'
                                                  '    "stopped": false,\n'
                                                  '    "circuit_tripped": false,\n'
                                                  '    "pool_broken": false\n'
                                                  '  },\n'
                                                  '  "error": null\n'
                                                  '}',
                                   'stderr': '',
                                   'stdout': '[rise_eo.download_cli] probing throughput '
                                             '(downloads 1 band file)…\n'
                                             '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                             '[rise_eo.download_cli] discovering + planning '
                                             'download…\n'
                                             '\n'
                                             '[rise_eo.download_cli] stopped: successful=5 '
                                             'failed=2 (total across passes 2) skipped=1 '
                                             '(2.35 GB) | transfer=40.0s convert=3.3s | '
                                             'probe=12.3 MB/s per-stream=58.6 wall=234.5 MB/s '
                                             '| stopped=True circuit_tripped=False '
                                             'pool_broken=False\n'},
 'failed_circuit_tripped': {'raised': None,
                            'rc': 1,
                            'result_text': '{\n'
                                           '  "step": "download-confirm-run",\n'
                                           '  "status": "failed",\n'
                                           '  "pass": 1,\n'
                                           '  "metrics": {\n'
                                           '    "successful": 5,\n'
                                           '    "failed": 0,\n'
                                           '    "failed_total": 0,\n'
                                           '    "skipped": 1,\n'
                                           '    "gb": 2.345,\n'
                                           '    "transfer_s": 40.0,\n'
                                           '    "transfer_wall_s": 10.0,\n'
                                           '    "convert_s": 3.3,\n'
                                           '    "probe_mb_per_s": 12.34,\n'
                                           '    "aggregate_mb_per_s": 58.62,\n'
                                           '    "wall_transfer_mb_per_s": 234.5,\n'
                                           '    "elapsed_s": 12.3,\n'
                                           '    "stopped": false,\n'
                                           '    "circuit_tripped": true,\n'
                                           '    "pool_broken": false\n'
                                           '  },\n'
                                           '  "expected": {\n'
                                           '    "failed": 0,\n'
                                           '    "stopped": false,\n'
                                           '    "circuit_tripped": false,\n'
                                           '    "pool_broken": false\n'
                                           '  },\n'
                                           '  "error": "circuit breaker tripped"\n'
                                           '}',
                            'stderr': '',
                            'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 '
                                      'band file)…\n'
                                      '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                      '[rise_eo.download_cli] discovering + planning download…\n'
                                      '\n'
                                      '[rise_eo.download_cli] failed: successful=5 failed=0 '
                                      '(total across passes 0) skipped=1 (2.35 GB) | '
                                      'transfer=40.0s convert=3.3s | probe=12.3 MB/s '
                                      'per-stream=58.6 wall=234.5 MB/s | stopped=False '
                                      'circuit_tripped=True pool_broken=False\n'},
 'failed_pool_broken_with_failures': {'raised': None,
                                      'rc': 1,
                                      'result_text': '{\n'
                                                     '  "step": "download-confirm-run",\n'
                                                     '  "status": "failed",\n'
                                                     '  "pass": 1,\n'
                                                     '  "metrics": {\n'
                                                     '    "successful": 5,\n'
                                                     '    "failed": 1,\n'
                                                     '    "failed_total": 1,\n'
                                                     '    "skipped": 1,\n'
                                                     '    "gb": 2.345,\n'
                                                     '    "transfer_s": 40.0,\n'
                                                     '    "transfer_wall_s": 10.0,\n'
                                                     '    "convert_s": 3.3,\n'
                                                     '    "probe_mb_per_s": 12.34,\n'
                                                     '    "aggregate_mb_per_s": 58.62,\n'
                                                     '    "wall_transfer_mb_per_s": 234.5,\n'
                                                     '    "elapsed_s": 12.3,\n'
                                                     '    "stopped": false,\n'
                                                     '    "circuit_tripped": false,\n'
                                                     '    "pool_broken": true\n'
                                                     '  },\n'
                                                     '  "expected": {\n'
                                                     '    "failed": 0,\n'
                                                     '    "stopped": false,\n'
                                                     '    "circuit_tripped": false,\n'
                                                     '    "pool_broken": false\n'
                                                     '  },\n'
                                                     '  "error": "1 file(s) failed on the '
                                                     'terminal pass; convert pool broken with '
                                                     'unresolved failures"\n'
                                                     '}',
                                      'stderr': '',
                                      'stdout': '[rise_eo.download_cli] probing throughput '
                                                '(downloads 1 band file)…\n'
                                                '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                                '[rise_eo.download_cli] discovering + planning '
                                                'download…\n'
                                                '\n'
                                                '[rise_eo.download_cli] failed: successful=5 '
                                                'failed=1 (total across passes 1) skipped=1 '
                                                '(2.35 GB) | transfer=40.0s convert=3.3s | '
                                                'probe=12.3 MB/s per-stream=58.6 wall=234.5 '
                                                'MB/s | stopped=False circuit_tripped=False '
                                                'pool_broken=True\n'},
 'failed_terminal_count': {'raised': None,
                           'rc': 1,
                           'result_text': '{\n'
                                          '  "step": "download-confirm-run",\n'
                                          '  "status": "failed",\n'
                                          '  "pass": 1,\n'
                                          '  "metrics": {\n'
                                          '    "successful": 2,\n'
                                          '    "failed": 3,\n'
                                          '    "failed_total": 3,\n'
                                          '    "skipped": 1,\n'
                                          '    "gb": 2.345,\n'
                                          '    "transfer_s": 40.0,\n'
                                          '    "transfer_wall_s": 10.0,\n'
                                          '    "convert_s": 3.3,\n'
                                          '    "probe_mb_per_s": 12.34,\n'
                                          '    "aggregate_mb_per_s": 58.62,\n'
                                          '    "wall_transfer_mb_per_s": 234.5,\n'
                                          '    "elapsed_s": 12.3,\n'
                                          '    "stopped": false,\n'
                                          '    "circuit_tripped": false,\n'
                                          '    "pool_broken": false\n'
                                          '  },\n'
                                          '  "expected": {\n'
                                          '    "failed": 0,\n'
                                          '    "stopped": false,\n'
                                          '    "circuit_tripped": false,\n'
                                          '    "pool_broken": false\n'
                                          '  },\n'
                                          '  "error": "3 file(s) failed on the terminal '
                                          'pass"\n'
                                          '}',
                           'stderr': '',
                           'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band '
                                     'file)…\n'
                                     '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                     '[rise_eo.download_cli] discovering + planning download…\n'
                                     '\n'
                                     '[rise_eo.download_cli] failed: successful=2 failed=3 (total '
                                     'across passes 3) skipped=1 (2.35 GB) | transfer=40.0s '
                                     'convert=3.3s | probe=12.3 MB/s per-stream=58.6 '
                                     'wall=234.5 MB/s | stopped=False circuit_tripped=False '
                                     'pool_broken=False\n'},
 'ok': {'raised': None,
        'rc': 0,
        'result_text': '{\n'
                       '  "step": "download-confirm-run",\n'
                       '  "status": "ok",\n'
                       '  "pass": 1,\n'
                       '  "metrics": {\n'
                       '    "successful": 5,\n'
                       '    "failed": 0,\n'
                       '    "failed_total": 0,\n'
                       '    "skipped": 1,\n'
                       '    "gb": 2.345,\n'
                       '    "transfer_s": 40.0,\n'
                       '    "transfer_wall_s": 10.0,\n'
                       '    "convert_s": 3.3,\n'
                       '    "probe_mb_per_s": 12.34,\n'
                       '    "aggregate_mb_per_s": 58.62,\n'
                       '    "wall_transfer_mb_per_s": 234.5,\n'
                       '    "elapsed_s": 12.3,\n'
                       '    "stopped": false,\n'
                       '    "circuit_tripped": false,\n'
                       '    "pool_broken": false\n'
                       '  },\n'
                       '  "expected": {\n'
                       '    "failed": 0,\n'
                       '    "stopped": false,\n'
                       '    "circuit_tripped": false,\n'
                       '    "pool_broken": false\n'
                       '  },\n'
                       '  "error": null\n'
                       '}',
        'stderr': '',
        'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band file)…\n'
                  '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                  '[rise_eo.download_cli] discovering + planning download…\n'
                  '\n'
                  '[rise_eo.download_cli] ok: successful=5 failed=0 (total across passes 0) '
                  'skipped=1 (2.35 GB) | transfer=40.0s convert=3.3s | probe=12.3 MB/s '
                  'per-stream=58.6 wall=234.5 MB/s | stopped=False circuit_tripped=False '
                  'pool_broken=False\n'},
 'ok_quiet_noprobe': {'raised': None,
                      'rc': 0,
                      'result_text': '{\n'
                                     '  "step": "download-confirm-run",\n'
                                     '  "status": "ok",\n'
                                     '  "pass": 1,\n'
                                     '  "metrics": {\n'
                                     '    "successful": 5,\n'
                                     '    "failed": 0,\n'
                                     '    "failed_total": 0,\n'
                                     '    "skipped": 1,\n'
                                     '    "gb": 2.345,\n'
                                     '    "transfer_s": 40.0,\n'
                                     '    "transfer_wall_s": 10.0,\n'
                                     '    "convert_s": 3.3,\n'
                                     '    "probe_mb_per_s": 0.0,\n'
                                     '    "aggregate_mb_per_s": 58.62,\n'
                                     '    "wall_transfer_mb_per_s": 234.5,\n'
                                     '    "elapsed_s": 12.3,\n'
                                     '    "stopped": false,\n'
                                     '    "circuit_tripped": false,\n'
                                     '    "pool_broken": false\n'
                                     '  },\n'
                                     '  "expected": {\n'
                                     '    "failed": 0,\n'
                                     '    "stopped": false,\n'
                                     '    "circuit_tripped": false,\n'
                                     '    "pool_broken": false\n'
                                     '  },\n'
                                     '  "error": null\n'
                                     '}',
                      'stderr': '',
                      'stdout': '\n'
                                '[rise_eo.download_cli] ok: successful=5 failed=0 (total across '
                                'passes 0) skipped=1 (2.35 GB) | transfer=40.0s convert=3.3s '
                                '| probe=0.0 MB/s per-stream=58.6 wall=234.5 MB/s | '
                                'stopped=False circuit_tripped=False pool_broken=False\n'},
 'ok_recovered_transient': {'raised': None,
                            'rc': 0,
                            'result_text': '{\n'
                                           '  "step": "download-confirm-run",\n'
                                           '  "status": "ok",\n'
                                           '  "pass": 2,\n'
                                           '  "metrics": {\n'
                                           '    "successful": 8,\n'
                                           '    "failed": 0,\n'
                                           '    "failed_total": 2,\n'
                                           '    "skipped": 1,\n'
                                           '    "gb": 2.445,\n'
                                           '    "transfer_s": 42.0,\n'
                                           '    "transfer_wall_s": 11.0,\n'
                                           '    "convert_s": 3.3,\n'
                                           '    "probe_mb_per_s": 12.34,\n'
                                           '    "aggregate_mb_per_s": 58.21,\n'
                                           '    "wall_transfer_mb_per_s": 222.27,\n'
                                           '    "elapsed_s": 12.3,\n'
                                           '    "stopped": false,\n'
                                           '    "circuit_tripped": false,\n'
                                           '    "pool_broken": false\n'
                                           '  },\n'
                                           '  "expected": {\n'
                                           '    "failed": 0,\n'
                                           '    "stopped": false,\n'
                                           '    "circuit_tripped": false,\n'
                                           '    "pool_broken": false\n'
                                           '  },\n'
                                           '  "error": null\n'
                                           '}',
                            'stderr': '',
                            'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 '
                                      'band file)…\n'
                                      '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                      '[rise_eo.download_cli] discovering + planning download…\n'
                                      '\n'
                                      '[rise_eo.download_cli] ok: successful=8 failed=0 (total '
                                      'across passes 2) skipped=1 (2.44 GB) | transfer=42.0s '
                                      'convert=3.3s | probe=12.3 MB/s per-stream=58.2 '
                                      'wall=222.3 MB/s | stopped=False circuit_tripped=False '
                                      'pool_broken=False\n'},
 'pool_broken_without_failures_is_ok': {'raised': None,
                                        'rc': 0,
                                        'result_text': '{\n'
                                                       '  "step": "download-confirm-run",\n'
                                                       '  "status": "ok",\n'
                                                       '  "pass": 1,\n'
                                                       '  "metrics": {\n'
                                                       '    "successful": 5,\n'
                                                       '    "failed": 0,\n'
                                                       '    "failed_total": 0,\n'
                                                       '    "skipped": 1,\n'
                                                       '    "gb": 2.345,\n'
                                                       '    "transfer_s": 40.0,\n'
                                                       '    "transfer_wall_s": 10.0,\n'
                                                       '    "convert_s": 3.3,\n'
                                                       '    "probe_mb_per_s": 12.34,\n'
                                                       '    "aggregate_mb_per_s": 58.62,\n'
                                                       '    "wall_transfer_mb_per_s": 234.5,\n'
                                                       '    "elapsed_s": 12.3,\n'
                                                       '    "stopped": false,\n'
                                                       '    "circuit_tripped": false,\n'
                                                       '    "pool_broken": true\n'
                                                       '  },\n'
                                                       '  "expected": {\n'
                                                       '    "failed": 0,\n'
                                                       '    "stopped": false,\n'
                                                       '    "circuit_tripped": false,\n'
                                                       '    "pool_broken": false\n'
                                                       '  },\n'
                                                       '  "error": null\n'
                                                       '}',
                                        'stderr': '',
                                        'stdout': '[rise_eo.download_cli] probing throughput '
                                                  '(downloads 1 band file)…\n'
                                                  '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                                  '[rise_eo.download_cli] discovering + planning '
                                                  'download…\n'
                                                  '\n'
                                                  '[rise_eo.download_cli] ok: successful=5 '
                                                  'failed=0 (total across passes 0) skipped=1 '
                                                  '(2.35 GB) | transfer=40.0s convert=3.3s | '
                                                  'probe=12.3 MB/s per-stream=58.6 wall=234.5 '
                                                  'MB/s | stopped=False circuit_tripped=False '
                                                  'pool_broken=True\n'},
 'real_expected_json': {'raised': None,
                        'rc': 0,
                        'result_text': '{\n'
                                       '  "step": "download-confirm-run",\n'
                                       '  "status": "ok",\n'
                                       '  "pass": 1,\n'
                                       '  "metrics": {\n'
                                       '    "successful": 5,\n'
                                       '    "failed": 0,\n'
                                       '    "failed_total": 0,\n'
                                       '    "skipped": 1,\n'
                                       '    "gb": 2.345,\n'
                                       '    "transfer_s": 40.0,\n'
                                       '    "transfer_wall_s": 10.0,\n'
                                       '    "convert_s": 3.3,\n'
                                       '    "probe_mb_per_s": 12.34,\n'
                                       '    "aggregate_mb_per_s": 58.62,\n'
                                       '    "wall_transfer_mb_per_s": 234.5,\n'
                                       '    "elapsed_s": 12.3,\n'
                                       '    "stopped": false,\n'
                                       '    "circuit_tripped": false,\n'
                                       '    "pool_broken": false\n'
                                       '  },\n'
                                       '  "expected": {\n'
                                       '    "failed": 99,\n'
                                       '    "stopped": false,\n'
                                       '    "circuit_tripped": false,\n'
                                       '    "pool_broken": false,\n'
                                       '    "custom": "x"\n'
                                       '  },\n'
                                       '  "error": null\n'
                                       '}',
                        'stderr': '',
                        'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band '
                                  'file)…\n'
                                  '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                                  '[rise_eo.download_cli] discovering + planning download…\n'
                                  '\n'
                                  '[rise_eo.download_cli] ok: successful=5 failed=0 (total across '
                                  'passes 0) skipped=1 (2.35 GB) | transfer=40.0s '
                                  'convert=3.3s | probe=12.3 MB/s per-stream=58.6 wall=234.5 '
                                  'MB/s | stopped=False circuit_tripped=False '
                                  'pool_broken=False\n'},
 'stopped': {'raised': None,
             'rc': 0,
             'result_text': '{\n'
                            '  "step": "download-confirm-run",\n'
                            '  "status": "stopped",\n'
                            '  "pass": 1,\n'
                            '  "metrics": {\n'
                            '    "successful": 2,\n'
                            '    "failed": 0,\n'
                            '    "failed_total": 0,\n'
                            '    "skipped": 1,\n'
                            '    "gb": 2.345,\n'
                            '    "transfer_s": 40.0,\n'
                            '    "transfer_wall_s": 10.0,\n'
                            '    "convert_s": 3.3,\n'
                            '    "probe_mb_per_s": 12.34,\n'
                            '    "aggregate_mb_per_s": 58.62,\n'
                            '    "wall_transfer_mb_per_s": 234.5,\n'
                            '    "elapsed_s": 12.3,\n'
                            '    "stopped": true,\n'
                            '    "circuit_tripped": false,\n'
                            '    "pool_broken": false\n'
                            '  },\n'
                            '  "expected": {\n'
                            '    "failed": 0,\n'
                            '    "stopped": false,\n'
                            '    "circuit_tripped": false,\n'
                            '    "pool_broken": false\n'
                            '  },\n'
                            '  "error": null\n'
                            '}',
             'stderr': '',
             'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band file)…\n'
                       '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                       '[rise_eo.download_cli] discovering + planning download…\n'
                       '\n'
                       '[rise_eo.download_cli] stopped: successful=2 failed=0 (total across '
                       'passes 0) skipped=1 (2.35 GB) | transfer=40.0s convert=3.3s | '
                       'probe=12.3 MB/s per-stream=58.6 wall=234.5 MB/s | stopped=True '
                       'circuit_tripped=False pool_broken=False\n'},
 'zero_seconds': {'raised': None,
                  'rc': 0,
                  'result_text': '{\n'
                                 '  "step": "download-confirm-run",\n'
                                 '  "status": "ok",\n'
                                 '  "pass": 1,\n'
                                 '  "metrics": {\n'
                                 '    "successful": 5,\n'
                                 '    "failed": 0,\n'
                                 '    "failed_total": 0,\n'
                                 '    "skipped": 0,\n'
                                 '    "gb": 0.005,\n'
                                 '    "transfer_s": 0.0,\n'
                                 '    "transfer_wall_s": 0.0,\n'
                                 '    "convert_s": 0.0,\n'
                                 '    "probe_mb_per_s": 12.34,\n'
                                 '    "aggregate_mb_per_s": 0.0,\n'
                                 '    "wall_transfer_mb_per_s": 0.0,\n'
                                 '    "elapsed_s": 12.3,\n'
                                 '    "stopped": false,\n'
                                 '    "circuit_tripped": false,\n'
                                 '    "pool_broken": false\n'
                                 '  },\n'
                                 '  "expected": {\n'
                                 '    "failed": 0,\n'
                                 '    "stopped": false,\n'
                                 '    "circuit_tripped": false,\n'
                                 '    "pool_broken": false\n'
                                 '  },\n'
                                 '  "error": null\n'
                                 '}',
                  'stderr': '',
                  'stdout': '[rise_eo.download_cli] probing throughput (downloads 1 band file)…\n'
                            '[rise_eo.download_cli] probe: 12.3 MB/s\n'
                            '[rise_eo.download_cli] discovering + planning download…\n'
                            '\n'
                            '[rise_eo.download_cli] ok: successful=5 failed=0 (total across '
                            'passes 0) skipped=0 (0.01 GB) | transfer=0.0s convert=0.0s | '
                            'probe=12.3 MB/s per-stream=0.0 wall=0.0 MB/s | stopped=False '
                            'circuit_tripped=False pool_broken=False\n'}}
