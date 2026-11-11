"""Score saved judgments and verify the blind pairs against the original answers."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent / "fresh50"

def read(name):
    return json.loads((root / (name + ".json")).read_text())

pairs, key = read("pairs"), read("key")
assert len(pairs) == len(key) == len({pair["prompt"] for pair in pairs}) == 25
for author in ("gemini", "qwen"):
    original = {row["prompt"]: row["text"] for row in read(author)}
    assert len(original) == len(pairs)
    for pair, answer in zip(pairs, key):
        label = answer if author == "gemini" else {"A": "B", "B": "A"}[answer]
        assert pair[label] == original[pair["prompt"]]
    guesses = read(author + "-guesses")
    assert len(guesses) == len(key) and set(guesses) <= {"A", "B"}
    print(f"{author}: {sum(a == b for a, b in zip(guesses, key))}/{len(key)} pairs")
