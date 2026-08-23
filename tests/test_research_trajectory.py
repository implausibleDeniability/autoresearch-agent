import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import chdir
from pathlib import Path

import pytest

from generate_trajectory import main as generate_main
from trajectory_data import (
    CURRENT_FIELDS,
    LEGACY_FIELDS,
    REPLAY_FIELDS,
    Experiment,
    IncumbentState,
    Trajectory,
    TrajectoryError,
    build_incumbent_states,
    load_trajectory,
    read_blind_result,
    read_experiments,
)
from trajectory_layout import wrap_text, x_ticks, y_domain
from trajectory_milestones import layout_milestones
from trajectory_svg import render_svg

ROOT = Path(__file__).parents[1]
SKILL = ROOT / ".agents" / "skills" / "generate-autoresearch-trajectory"
SCRIPT = SKILL / "scripts" / "generate_trajectory.py"
EXAMPLES = SKILL / "examples"
FIXTURES = ROOT / "tests" / "fixtures"


def test_example_generates_accessible_deterministic_svg(tmp_path):
    # setup
    output = tmp_path / "trajectory.svg"
    command = [
        sys.executable,
        str(SCRIPT),
        "--results",
        str(EXAMPLES / "results.tsv"),
        "--run-log",
        str(EXAMPLES / "example-test-result.txt"),
        "--output",
        str(output),
    ]

    # operate
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    first_bytes = output.read_bytes()
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    direct = render_svg(
        load_trajectory(EXAMPLES / "results.tsv", run_log_path=EXAMPLES / "example-test-result.txt"),
        heading="PII extraction research trajectory",
    )

    # check
    assert first.returncode == 0
    assert second.returncode == 0
    assert output.read_bytes() == first_bytes
    assert "Development experiments: 20" in first.stdout
    assert "Experiment 9 · +2.7 pp · Deterministic email recovery" in first.stdout
    root = ET.fromstring(first_bytes)
    assert root.attrib["role"] == "img"
    assert root.attrib["aria-labelledby"] == "trajectory-title trajectory-desc"
    assert root.find("{http://www.w3.org/2000/svg}title").text == "PII extraction research trajectory"
    description = root.find("{http://www.w3.org/2000/svg}desc").text
    assert "finding: Confirmed the email gain on a repeat" in description
    text = " ".join(root.itertext())
    assert "WHAT WORKED" in text
    assert "Test dataset: 84.8%" in text
    assert "OCR-aware name variants" in text
    assert root.find(".//*[@data-series='dev-dataset']") is not None
    assert root.find(".//*[@data-series='milestone-label']") is not None
    assert root.find(".//*[@data-score='0.848000']") is not None
    assert direct.encode() == first_bytes


def test_real_long_milestones_render_complete_without_overlap():
    trajectory = load_trajectory(
        FIXTURES / "research_trajectory_long_text.tsv",
        run_log_path=FIXTURES / "research_trajectory_test_result.txt",
    )

    layout = layout_milestones(trajectory.milestones, top=164)
    svg = render_svg(trajectory, heading="PII extraction research trajectory")
    root = ET.fromstring(svg)
    milestone_group = root.find(".//*[@data-section='milestones']")
    visible_text = " ".join(" ".join(milestone_group.itertext()).split())
    height = float(root.attrib["viewBox"].split()[-1])

    assert all(state.description in visible_text for state in trajectory.milestones)
    assert all(state.finding in visible_text for state in trajectory.milestones)
    assert "…" not in visible_text
    assert all(left.bottom <= right.top for left, right in zip(layout.rows, layout.rows[1:]))
    assert layout.bottom + 40 <= height


def test_milestone_number_and_metadata_share_an_optical_top_edge():
    trajectory = load_trajectory(FIXTURES / "research_trajectory_long_text.tsv")

    root = ET.fromstring(render_svg(trajectory, heading="Trajectory"))
    numbers = root.findall(".//*[@data-role='milestone-number']")
    metadata = root.findall(".//*[@data-role='milestone-meta']")

    assert len(numbers) == len(metadata) == 4
    for number, meta in zip(numbers, metadata):
        number_top = float(number.attrib["y"]) - float(number.attrib["font-size"]) * 0.82
        meta_top = float(meta.attrib["y"]) - float(meta.attrib["font-size"]) * 0.82
        assert abs(number_top - meta_top) <= 1


def test_seven_long_milestones_expand_canvas_without_losing_text():
    title = "accepted recovery strategy for dense correspondence and citation records"
    finding = (
        "Confirmed the representative improvement without unrelated false positives or "
        "increased evaluation cost"
    )
    baseline = IncumbentState(
        experiment=1,
        commit="baseline",
        score=0.700,
        description="Baseline",
        finding="Baseline",
        delta=None,
    )
    milestones = tuple(
        IncumbentState(
            experiment=index * 6,
            commit=f"commit{index}",
            score=0.700 + index * 0.025,
            description=title,
            finding=finding,
            delta=0.025,
        )
        for index in range(1, 8)
    )
    trajectory = Trajectory(
        experiment_count=42,
        states=(baseline, *milestones),
        milestones=milestones,
        blind=None,
    )

    layout = layout_milestones(trajectory.milestones, top=164)
    root = ET.fromstring(render_svg(trajectory, heading="Seven milestones"))
    visible_text = " ".join(" ".join(root.find(".//*[@data-section='milestones']").itertext()).split())

    assert float(root.attrib["viewBox"].split()[-1]) > 720
    assert all(left.bottom <= right.top for left, right in zip(layout.rows, layout.rows[1:]))
    assert visible_text.count(title) == 7
    assert visible_text.count(finding) == 7
    assert "…" not in visible_text


def test_incumbent_episodes_preserve_repeats_returns_and_downward_steps():
    experiments = [
        _experiment(1, "aaaaaaa", 0.700, status="keep"),
        _experiment(2, "bbbbbbb", 0.900, status="discard"),
        _experiment(3, "aaaaaaa", 0.720, status="keep"),
        _experiment(4, "bbbbbbb", 0.750, status="keep"),
        _experiment(5, "bbbbbbb", 0.770, status="keep"),
        _experiment(6, "ccccccc", 0.740, status="keep"),
        _experiment(7, "aaaaaaa", 0.800, status="keep"),
        _experiment(8, "ddddddd", 0.000, status="crash", dataset="debug"),
    ]

    states = build_incumbent_states(experiments)

    assert [state.experiment for state in states] == [3, 5, 6, 7]
    assert [state.score for state in states] == pytest.approx([0.710, 0.760, 0.740, 0.800])
    assert [state.delta for state in states[1:]] == pytest.approx([0.050, -0.020, 0.060])
    trajectory = _trajectory_file(experiments)
    svg = render_svg(trajectory, heading="Trajectory")
    assert 'data-experiment="6"' in svg
    assert 'data-score="0.740000"' in svg
    path = ET.fromstring(svg).find(".//*[@data-series='dev-dataset']")
    assert path.attrib["d"].endswith("H 772")


def test_baseline_repetitions_group_without_losing_experiment_one_origin(tmp_path):
    results = tmp_path / "results.tsv"
    _write_results(
        results,
        [
            _experiment(1, "aaaaaaa", 0.700, status="keep"),
            _experiment(2, "trial00", 0.710, status="inconclusive", dataset="dev-19k"),
            _experiment(3, "aaaaaaa", 0.720, status="keep"),
            _experiment(4, "bbbbbbb", 0.750, status="keep"),
        ],
    )

    trajectory = load_trajectory(results)
    svg = render_svg(trajectory, heading="Repeated baseline")

    assert [state.experiment for state in trajectory.states] == [3, 4]
    assert trajectory.states[0].score == pytest.approx(0.710)
    path = ET.fromstring(svg).find(".//*[@data-series='dev-dataset']")
    assert path.attrib["d"].startswith("M 88 ")


def test_milestones_select_top_seven_then_render_chronologically(tmp_path):
    score = 0.100
    rows = [_experiment(1, "base000", score, status="keep")]
    for number in range(2, 11):
        score += (number - 1) / 100
        rows.append(_experiment(number, f"commit{number}", score, status="keep"))
    results = tmp_path / "results.tsv"
    _write_results(results, rows)

    trajectory = load_trajectory(results)

    assert [state.experiment for state in trajectory.milestones] == list(range(4, 11))
    assert len(trajectory.milestones) == 7


def test_legacy_schema_falls_back_to_description_and_escapes_xml(tmp_path):
    results = tmp_path / "results.tsv"
    rows = [
        _experiment(1, "aaaaaaa", 0.700, status="keep", description="Base & <start>"),
        _experiment(2, "bbbbbbb", 0.760, status="keep", description="Names & emails <fixed>"),
    ]
    _write_results(results, rows, fields=LEGACY_FIELDS)

    trajectory = load_trajectory(results)
    svg = render_svg(trajectory, heading="A & B <research>")

    assert trajectory.milestones[0].finding == "Names & emails <fixed>"
    assert "A &amp; B &lt;research&gt;" in svg
    assert "Names &amp; emails &lt;fixed&gt;" in svg
    ET.fromstring(svg)


def test_replay_schema_preserves_evaluation_mode_and_legacy_defaults_live(tmp_path):
    replay_results = tmp_path / "replay.tsv"
    legacy_results = tmp_path / "legacy.tsv"
    _write_results(
        replay_results,
        [_experiment(1, "aaaaaaa", 0.700, status="keep", evaluation_mode="cached")],
        fields=REPLAY_FIELDS,
    )
    _write_results(
        legacy_results,
        [_experiment(1, "aaaaaaa", 0.700, status="keep")],
        fields=LEGACY_FIELDS,
    )

    replay = read_experiments(replay_results)
    legacy = read_experiments(legacy_results)

    assert replay[0].evaluation_mode == "cached"
    assert legacy[0].evaluation_mode == "live"


def test_replay_schema_rejects_unsupported_evaluation_mode(tmp_path):
    results = tmp_path / "results.tsv"
    experiment = _experiment(1, "aaaaaaa", 0.700, status="keep")
    values = _row_values(experiment)
    values["evaluation_mode"] = "sometimes"
    results.write_text(
        "\t".join(REPLAY_FIELDS) + "\n" + "\t".join(values[field] for field in REPLAY_FIELDS) + "\n"
    )

    with pytest.raises(TrajectoryError, match="unsupported evaluation_mode"):
        read_experiments(results)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("score", "nan", "score must be finite"),
        ("precision", "1.1", "precision must be finite"),
        ("cost", "-0.1", "cost must be finite"),
        ("status", "accepted", "unsupported status"),
        ("dataset", "test-hidden", "unsupported dataset"),
        ("finding", "", "accepted dev-205k finding must not be empty"),
    ],
)
def test_invalid_result_fields_fail_without_echoing_rows(tmp_path, field, value, message):
    results = tmp_path / "results.tsv"
    row = _row_values(_experiment(1, "secret1", 0.700, status="keep"))
    row[field] = value
    results.write_text(
        "\t".join(CURRENT_FIELDS) + "\n" + "\t".join(row[name] for name in CURRENT_FIELDS) + "\n"
    )

    with pytest.raises(TrajectoryError, match=message) as raised:
        read_experiments(results)

    assert "secret1" not in str(raised.value)


def test_reordered_header_and_missing_accepted_state_fail_clearly(tmp_path):
    reordered = tmp_path / "reordered.tsv"
    fields = list(CURRENT_FIELDS)
    fields[0], fields[1] = fields[1], fields[0]
    reordered.write_text("\t".join(fields) + "\n")
    with pytest.raises(TrajectoryError, match="exact 9-, 10-, or 11-column supported header"):
        read_experiments(reordered)

    results = tmp_path / "results.tsv"
    _write_results(results, [_experiment(1, "aaaaaaa", 0.700, status="discard")])
    with pytest.raises(TrajectoryError, match="run a complete dev-205k baseline"):
        load_trajectory(results)

    late_baseline = tmp_path / "late-baseline.tsv"
    _write_results(
        late_baseline,
        [
            _experiment(1, "aaaaaaa", 0.700, status="discard", dataset="dev-19k"),
            _experiment(2, "bbbbbbb", 0.750, status="keep"),
        ],
    )
    with pytest.raises(TrajectoryError, match="experiment 1 must be an accepted dev-205k baseline"):
        load_trajectory(late_baseline)


def test_blind_log_supports_absent_development_and_complete_shapes(tmp_path):
    development = tmp_path / "development.log"
    development.write_text("result_status=complete\nf_score=0.700000\n")
    assert read_blind_result(development) is None

    complete = tmp_path / "blind.log"
    complete.write_text(
        "ignored=safe\nf_score=0.810000\nprecision=0.900000\nrecall=0.795000\n"
        "api_cost_usd=0.004000\nduration_seconds=12.500000\n"
    )
    blind = read_blind_result(complete)
    assert blind.score == pytest.approx(0.81)
    assert blind.api_cost_usd == pytest.approx(0.004)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("f_score=0.8\n", "partial Test dataset result"),
        (
            "f_score=0.8\nf_score=0.9\nprecision=0.9\nrecall=0.8\napi_cost_usd=0.1\nduration_seconds=1\n",
            "duplicate f_score field",
        ),
        (
            "f_score=inf\nprecision=0.9\nrecall=0.8\napi_cost_usd=0.1\nduration_seconds=1\n",
            "f_score must be finite",
        ),
    ],
)
def test_blind_log_rejects_partial_duplicate_and_invalid_values(tmp_path, contents, message):
    run_log = tmp_path / "run.log"
    run_log.write_text(contents)

    with pytest.raises(TrajectoryError, match=message):
        read_blind_result(run_log)


def test_blind_log_rejects_invalid_utf8_without_leaking_content(tmp_path):
    run_log = tmp_path / "run.log"
    run_log.write_bytes(b"f_score=\xff\n")

    with pytest.raises(TrajectoryError, match="cannot read"):
        read_blind_result(run_log)


def test_results_and_title_reject_invalid_xml_text_boundaries(tmp_path):
    results = tmp_path / "results.tsv"
    results.write_bytes(b"\xff")
    with pytest.raises(TrajectoryError, match="invalid UTF-8"):
        read_experiments(results)

    trajectory = _trajectory_file([_experiment(1, "aaaaaaa", 0.700, status="keep")])
    with pytest.raises(TrajectoryError, match="title contains an illegal XML character"):
        render_svg(trajectory, heading="Unsafe\x01title")


def test_cli_reports_missing_explicit_log_and_missing_output_parent(tmp_path):
    results = tmp_path / "results.tsv"
    _write_results(results, [_experiment(1, "aaaaaaa", 0.700, status="keep")])
    missing_log = subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(results), "--run-log", str(tmp_path / "missing.log")],
        check=False,
        capture_output=True,
        text=True,
    )
    missing_parent = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results",
            str(results),
            "--output",
            str(tmp_path / "missing" / "result.svg"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing_log.returncode == 2
    assert "provide an existing file or use --no-run-log" in missing_log.stderr
    assert "Traceback" not in missing_log.stderr
    assert missing_parent.returncode == 2
    assert "output directory does not exist" in missing_parent.stderr
    assert not (tmp_path / "missing" / "result.svg").exists()


def test_direct_cli_defaults_replace_output_and_report_empty_milestones(tmp_path, capsys):
    results = tmp_path / "results.tsv"
    _write_results(results, [_experiment(1, "aaaaaaa", 0.700, status="keep")])
    output = tmp_path / "research-trajectory.svg"
    output.write_text("old")

    result = generate_main(["--results", str(results), "--title", "Custom title"])

    captured = capsys.readouterr()
    assert result == 0
    assert output.read_text().startswith("<?xml")
    assert "Custom title" in output.read_text()
    assert "Test dataset: not present" in captured.out
    assert "Selected milestones:\n  none" in captured.out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--results", "missing.tsv"], "cannot read missing.tsv"),
        (["--results", "results.tsv", "--run-log", "missing.log"], "explicit run log does not exist"),
        (["--results", "results.tsv", "--output", "result.png"], "output must use the .svg extension"),
        (
            ["--results", "results.tsv", "--output", "missing/result.svg"],
            "output directory does not exist",
        ),
    ],
)
def test_direct_cli_returns_actionable_errors(tmp_path, capsys, arguments, message):
    _write_results(tmp_path / "results.tsv", [_experiment(1, "aaaaaaa", 0.700, status="keep")])

    with chdir(tmp_path):
        result = generate_main(arguments)

    captured = capsys.readouterr()
    assert result == 2
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_direct_cli_infers_complete_blind_log(tmp_path, capsys):
    results = tmp_path / "results.tsv"
    _write_results(
        results,
        [
            _experiment(1, "aaaaaaa", 0.700, status="keep"),
            _experiment(2, "bbbbbbb", 0.750, status="keep"),
        ],
    )
    (tmp_path / "run.log").write_text(
        "f_score=0.740000\nprecision=0.800000\nrecall=0.730000\n"
        "api_cost_usd=0.004000\nduration_seconds=12.500000\n"
    )

    result = generate_main(["--results", str(results)])

    assert result == 0
    assert "Test dataset: 74.0%" in capsys.readouterr().out
    assert 'data-score="0.740000"' in (tmp_path / "research-trajectory.svg").read_text()


def test_direct_cli_can_ignore_inferred_blind_log(tmp_path, capsys):
    results = tmp_path / "results.tsv"
    _write_results(results, [_experiment(1, "aaaaaaa", 0.700, status="keep")])
    (tmp_path / "run.log").write_text("f_score=0.740000\n")

    result = generate_main(["--results", str(results), "--no-run-log"])

    assert result == 0
    assert "Test dataset: not present" in capsys.readouterr().out


def test_single_state_without_blind_has_stable_empty_milestone_layout(tmp_path):
    results = tmp_path / "results.tsv"
    _write_results(results, [_experiment(1, "aaaaaaa", 0.700, status="keep")])

    trajectory = load_trajectory(results)
    svg = render_svg(trajectory, heading="One state")

    assert trajectory.milestones == ()
    assert "No accepted improvement yet" in svg
    assert "Test dataset" not in svg
    assert "TEST" not in svg
    assert 'd="M 88 367 H 772"' in svg


def test_clustered_milestones_omit_overlapping_chart_labels():
    score = 0.700
    experiments = [_experiment(1, "base000", score, status="keep")]
    for number in range(2, 9):
        score += 0.015
        experiments.append(_experiment(number, f"commit{number}", score, status="keep"))
    for number in range(9, 23):
        experiments.append(_experiment(number, f"discard{number}", score, status="discard"))

    svg = render_svg(_trajectory_file(experiments), heading="Clustered trajectory")

    assert 'data-series="milestone-label"' not in svg
    assert "EXP 08" in svg
    assert "07" in svg


def test_layout_helpers_bound_ticks_and_truncate_long_unicode():
    lines = wrap_text("Оченьдлинноесловобезпробелов и короткий вывод", width=12, max_lines=2)
    lower, upper, ticks = y_domain([0.742, 0.742])

    assert len(lines) == 2
    assert all(len(line) <= 12 for line in lines)
    assert lines[0].endswith("…")
    assert 0 <= lower < 0.742 < upper <= 1
    assert 3 <= len(ticks) <= 7
    assert len(x_ticks(40)) <= 7
    assert x_ticks(1) == (1,)
    assert wrap_text("abcdefghijkl short", width=8, max_lines=2) == ("abcdefg…", "short")


def test_results_errors_cover_empty_rows_columns_numbers_and_xml_controls(tmp_path):
    empty = tmp_path / "empty.tsv"
    empty.write_text("\t".join(CURRENT_FIELDS) + "\n")
    with pytest.raises(TrajectoryError, match="no experiment rows"):
        read_experiments(empty)

    short = tmp_path / "short.tsv"
    short.write_text("\t".join(CURRENT_FIELDS) + "\nonly\tone\n")
    with pytest.raises(TrajectoryError, match="column count does not match"):
        read_experiments(short)

    not_a_number = tmp_path / "number.tsv"
    row = _row_values(_experiment(1, "aaaaaaa", 0.700, status="keep"))
    row["score"] = "many"
    not_a_number.write_text(
        "\t".join(CURRENT_FIELDS) + "\n" + "\t".join(row[field] for field in CURRENT_FIELDS) + "\n"
    )
    with pytest.raises(TrajectoryError, match="score must be a number"):
        read_experiments(not_a_number)

    illegal = tmp_path / "illegal.tsv"
    row["score"] = "0.700000"
    row["description"] = "bad\x00text"
    illegal.write_text(
        "\t".join(CURRENT_FIELDS) + "\n" + "\t".join(row[field] for field in CURRENT_FIELDS) + "\n"
    )
    with pytest.raises(TrajectoryError, match="illegal XML character"):
        read_experiments(illegal)


def test_results_treat_quotes_as_plain_tsv_text_without_swallowing_later_rows(tmp_path):
    results = tmp_path / "results.tsv"
    first = _row_values(_experiment(1, "aaaaaaa", 0.700, status="keep"))
    first["finding"] = 'A literal "quote does not start a multiline field'
    second = _row_values(_experiment(2, "bbbbbbb", 0.750, status="keep"))
    results.write_text(
        "\t".join(CURRENT_FIELDS)
        + "\n"
        + "\t".join(first[field] for field in CURRENT_FIELDS)
        + "\n"
        + "\t".join(second[field] for field in CURRENT_FIELDS)
        + "\n"
    )

    experiments = read_experiments(results)

    assert len(experiments) == 2
    assert experiments[0].finding == first["finding"]


def _experiment(
    number: int,
    commit: str,
    score: float,
    *,
    status: str,
    dataset: str = "dev-205k",
    description: str | None = None,
    evaluation_mode: str = "live",
) -> Experiment:
    title = description or f"Experiment {number}"
    return Experiment(
        number,
        commit,
        score,
        min(1, score + 0.05),
        max(0, score - 0.01),
        0.2,
        status,
        title,
        dataset,
        0.01,
        f"Finding {number}",
        evaluation_mode,
    )


def _write_results(path: Path, experiments: list[Experiment], *, fields=CURRENT_FIELDS) -> None:
    lines = ["\t".join(fields)]
    for experiment in experiments:
        values = _row_values(experiment)
        lines.append("\t".join(values[field] for field in fields))
    path.write_text("\n".join(lines) + "\n")


def _row_values(experiment: Experiment) -> dict[str, str]:
    return {
        "commit": experiment.commit,
        "score": f"{experiment.score:.6f}",
        "precision": f"{experiment.precision:.6f}",
        "recall": f"{experiment.recall:.6f}",
        "cost": f"{experiment.cost:.6f}",
        "status": experiment.status,
        "description": experiment.description,
        "dataset": experiment.dataset,
        "budget_cost_usd": f"{experiment.budget_cost_usd:.6f}",
        "finding": experiment.finding,
        "evaluation_mode": experiment.evaluation_mode,
    }


def _trajectory_file(experiments: list[Experiment]) -> Trajectory:
    states = build_incumbent_states(experiments)
    milestones = tuple(state for state in states[1:] if state.delta and state.delta > 0)
    return Trajectory(len(experiments), tuple(states), milestones, None)
