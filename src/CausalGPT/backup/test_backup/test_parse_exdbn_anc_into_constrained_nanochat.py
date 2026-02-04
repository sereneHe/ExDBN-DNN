import pytest
from pathlib import Path
from CausalGPT.constrained_nanochat import _normalize_anc_text

def test_parse_exdbn_anc_into_constrained_nanochat():
    anc_path = Path("/Users/xiaoyuhe/EXDBN-LLM/ExDBN-DNN/reports/causalgpt_runs_codiet/codiet/codiet_302_0_conf0.99999/ExDBN_LLM.anc")
    anc_text = anc_path.read_text()
    norm_text, arcs = _normalize_anc_text(anc_text)
    print("\n===== Normalized ANC TEXT =====\n")
    print(norm_text)
    print(f"\nTotal arcs parsed: {len(arcs)}")
    print("Sample arcs:", arcs[:5])
    assert len(arcs) > 0
