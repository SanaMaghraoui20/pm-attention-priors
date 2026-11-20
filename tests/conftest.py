import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def toy_log():
    """60 synthetic cases of a small loop-and-choice process, as a data dict."""
    rng = random.Random(0)
    acts = ["Register", "Assign", "Work", "Wait", "Resolve", "Close"]
    item2id = {"<PAD>": 0, "<MASK>": 1, **{a: k + 2 for k, a in enumerate(acts)}}
    seqs = {}
    for c in range(60):
        s = ["Register", "Assign"]
        for _ in range(rng.randint(1, 4)):
            s.append(rng.choice(["Work", "Wait"]))
        s += ["Resolve", "Close"]
        seqs[c] = [item2id[a] for a in s]
    return {"sequences": seqs, "item2id": item2id,
            "id2item": {v: k for k, v in item2id.items()},
            "n_items": len(item2id), "n_users": len(seqs)}
