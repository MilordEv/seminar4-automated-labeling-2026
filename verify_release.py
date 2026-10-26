"""Быстрая проверка данных и сохранённых результатов, без GPU и инференса."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "seminar.ipynb").read_text())
data = pd.read_parquet(ROOT / "artifacts/dataset_prepared.parquet")
pred = pd.read_parquet(ROOT / "artifacts/model_predictions.parquet")
humans = pd.read_parquet(ROOT / "artifacts/human_answers.parquet")
ensemble = pd.read_parquet(ROOT / "artifacts/ensemble_predictions.parquet")
assert data.groupby("split").size().to_dict() == {"calibration": 50, "prompt_examples": 60, "test": 150}
assert data.task_id.is_unique and pred.task_id.is_unique
gold = data.set_index("task_id")
assert pred.golden_label.tolist() == gold.loc[pred.task_id, "golden_label"].tolist()
assert pred.split.tolist() == gold.loc[pred.task_id, "split"].tolist()
assert pred.confidence.between(0, 1).all()
assert set(humans.columns) == {"task_id", "performer_id", "label", "status"}
assert humans.performer_id.str.fullmatch(r"worker_\d+").all()
assert ensemble.groupby("model_name").size().tolist() == [200, 200, 200]
for stage in ["prompt_comparison", "few_shot", "thinking", "output_format", "winner", "confidence"]:
    paths = list((ROOT / "artifacts/hf_cache").glob(stage + "_*.json"))
    assert len(paths) == 1, (stage, paths)
    frame = pd.read_json(paths[0], orient="table", precise_float=True)
    if stage == "confidence":
        assert frame.set_index("task_id").loc[pred.task_id, "confidence"].tolist() == pred.confidence.tolist()
    if stage == "winner":
        assert frame.set_index("task_id").loc[pred.task_id, "raw"].tolist() == pred.raw.tolist()
    if stage == "few_shot":
        assert sorted(frame["Примеров"].unique()) == [0, 1, 3, 5, 10, 20]
        assert frame.groupby(["Способ", "Примеров"]).size().eq(50).all()
for c in nb["cells"]:
    source = "".join(c["source"])
    if c["cell_type"] == "code" and not any(l.lstrip().startswith(("!", "%")) for l in c["source"]):
        ast.parse(source)
    assert not any(o.get("output_type") == "error" for o in c.get("outputs", []))
    assert "seminar4-model-as-annotator-data" not in source
    assert "USD" not in source
ledger = pd.read_csv(ROOT / "artifacts/leaderboard.csv")
np.testing.assert_allclose(ledger.cost, ledger.generation_cost + ledger.confidence_cost + ledger.human_cost)
assert (ledger.loc[ledger.human_assignments == 0, "cost"] > 0).all()
assert np.isclose(ledger.loc[ledger.scheme == "Один человек (среднее)", "cost"].iloc[0], 0.3)
test = pred[pred.split == "test"]
model_row = ledger[ledger.scheme == "Модель: thinking"].iloc[0]
assert np.isclose(model_row.accuracy, accuracy_score(test.golden_label, test.label))
assert np.isclose(model_row.macro_f1, f1_score(test.golden_label, test.label,
    labels=sorted(data.golden_label.unique()), average="macro", zero_division=0))
print("PASS: splits, pseudonyms, six caches, predictions, confidence, notebook syntax and pipeline costs")
