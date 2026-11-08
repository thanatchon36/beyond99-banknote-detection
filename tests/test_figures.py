from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def test_collision_check_flags_overlapping_text(mf):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "first label", transform=ax.transAxes)
    ax.text(0.5, 0.5, "second label", transform=ax.transAxes)
    assert any("text/text" in i for i in mf.check(fig, "overlap"))
    plt.close(fig)


def test_collision_check_flags_text_on_data(mf):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([0.5], [0.5])
    ax.text(0.5, 0.5, "on the point", ha="center", va="center")
    assert any("text/data" in i for i in mf.check(fig, "on-data"))
    plt.close(fig)


def test_paper_tables_regenerate_from_results(mf, lofo):
    mf.tables(lofo.reset_index())
    for name in ("e1.tex", "e3.tex", "lofo.tex"):
        assert (Path(mf.TABDIR) / name).read_text() == (PAPER / "tables" / name).read_text(), name
