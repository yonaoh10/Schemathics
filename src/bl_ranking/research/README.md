# Vendored research code — do not edit

These two files are copied byte-for-byte from
`DYK-Schemathics/Candidates@dev` (`ae1fce15d40a62cf2c5ce1b2b0b7bd1b979d8707`,
2026-08-23). They define what the models are and what the features mean, and the brief
says not to change their functions and logic.

```
b1274c55fb6b8eff0aad333ca540aba53e991a6862edefd9eaa434e5b43a573c  bl_models_train.py
d9ac9696ede0404c0f5273903394f085e2cb1b538187593afd0f1cb0b332b911  bl_exp_payout_predictor.py
```

Nothing in this directory is imported directly by production code paths. Everything
around it subclasses:

| production class | subclasses | overrides |
|---|---|---|
| `training.trainer.ProductionTrainer` | `BLPayoutModelsFit` | the payout estimator source; metric capture (calls `super()` first) |
| `serving.research_path.ServingPredictor` | `BLPayoutModelsPredict` | logging, warning capture, model loading, gender lookup, and `import_preprocess` |

Every feature-engineering method is **inherited unchanged**.

`training.job.research_code_sha()` hashes both files on every run and records the digest
as an MLflow tag, and `tests/test_pipeline.py::test_research_code_checksum_detects_an_edit`
proves the check works. So "the given scripts were not modified" is a fact on every run,
not a claim in a README.

Two things to know if you read them:

- They only parse on **Python 3.12+**. Lines 263-265 of `bl_models_train.py` use PEP 701
  f-strings (a newline and a nested same-type quote inside the expression), which is a
  `SyntaxError` on 3.11.
- Importing either one executes `nd = NameDataset()` at module scope: 9.5 s and 2.1 GB
  resident. That is why the serving path never imports `bl_exp_payout_predictor` unless
  `serving.feature_path = research` is set explicitly. See `models/gender_lut.py`.
