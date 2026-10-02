"""Lane Q5: the v2 cells figure covers every seed cell once and its x tick labels do not overlap."""
import csv

from tools.experiment_b2_seeds import figures_v2


def test_order_covers_seed_cells():
    with (figures_v2.SUMMARY / "seed-cells.csv").open() as stream:
        data = list(csv.DictReader(stream))
    for model in figures_v2.MODELS:
        cells = {(r["format"], r["recipe"]) for r in data if r["model"] == model}
        assert cells == set(figures_v2.ORDER)
    assert len(figures_v2.ORDER) == len(set(figures_v2.ORDER))
    assert figures_v2.GROUPS[0][0] == 0 and figures_v2.GROUPS[-1][1] == len(figures_v2.ORDER)


def test_v2_layout_has_no_overlapping_ticks(tmp_path):
    figures_v2.cells_figure_v2(str(figures_v2.SUMMARY), str(tmp_path), partial=True)  # raises on overlap
    assert (tmp_path / "b2-seeds-cells-v2.png").stat().st_size > 0
