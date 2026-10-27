"""Dataset adapters: external datasets → the ``cases.jsonl`` contract (§2).

Each adapter converts one real dataset into the case shape the pipeline expects:
``{case_id, text, gold, strata}``. Keeping them here (not in fixture.py) keeps the
synthetic fixture and real-data path clearly separate.
"""
