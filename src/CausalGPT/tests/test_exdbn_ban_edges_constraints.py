import numpy as np

from CausalGPT import exdbn_ban_edges


def test_write_camml_anc_format(tmp_path):
    anc_path = tmp_path / "t.anc"
    exdbn_ban_edges.write_camml_anc(
        anc_path,
        n_nodes=3,
        conf=0.99999,
        anc=[(0, 2)],
        forb_anc=[(1, 2)],
        abs_edges=[(2, 1)],
    )

    text = anc_path.read_text()
    assert text.startswith("arcs{\n")
    assert "0 => 2 0.99999;" in text
    assert "1 => 2" in text  # reconf line (1-conf)
    assert "2 -> 1" in text  # hard edge forbid
    assert text.rstrip().endswith("}")


def test_evaluate_constraints_counts():
    # 0 -> 1 -> 2
    adj = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=int,
    )

    metrics = exdbn_ban_edges.evaluate_constraints(
        adj,
        anc=[(0, 2)],
        forb_anc=[(2, 0)],
        abs_edges=[(0, 2)],
    )

    assert metrics["soft_anc_satisfied"] == 1  # 0 reaches 2
    assert metrics["soft_forb_satisfied"] == 1  # 2 does not reach 0
    assert metrics["hard_violations"] == 0  # direct edge 0->2 not present
