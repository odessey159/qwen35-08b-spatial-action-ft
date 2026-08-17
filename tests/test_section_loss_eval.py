from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.evaluate_section_losses import (
    MetricAccumulator,
    materialize_checkpoint_view,
    section_body_char_span,
    select_rows,
)


class SectionLossEvalTests(unittest.TestCase):
    def test_section_body_span_excludes_tags_and_surrounding_whitespace(self) -> None:
        content = "<action>\nGotoLocation(Desk)\nPickupObject(Book)\n</action>\n"
        start, end = section_body_char_span("action", content)
        stripped = content.strip()
        self.assertEqual(
            stripped[start:end],
            "GotoLocation(Desk)\nPickupObject(Book)",
        )

    def test_metric_accumulator_reports_micro_loss_and_accuracy(self) -> None:
        metric = MetricAccumulator()
        metric.update([1.0, 2.0], [True, False])
        metric.update([3.0], [True])
        self.assertEqual(metric.token_count, 3)
        self.assertAlmostEqual(metric.loss, 2.0)
        self.assertAlmostEqual(metric.token_accuracy, 2 / 3)
        self.assertAlmostEqual(metric.exact_match, 0.5)

    def test_select_rows_is_deterministic_and_keeps_source_indices(self) -> None:
        rows = [{"value": index} for index in range(20)]
        first = select_rows(rows, 5, 42)
        second = select_rows(rows, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual([index for index, _ in first], sorted(index for index, _ in first))

    def test_materialize_checkpoint_strips_sync_suffix_with_hardlink_or_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "checkpoint-1"
            source.mkdir()
            for name in ("config.json", "model.safetensors", "tokenizer.json"):
                (source / f"{name}.codex_sync_part").write_text(name, encoding="utf-8")
            view = materialize_checkpoint_view(source, root / "view")
            for name in ("config.json", "model.safetensors", "tokenizer.json"):
                self.assertEqual((view / name).read_text(encoding="utf-8"), name)

    def test_materialize_accepts_a_sharded_safetensors_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "base-model"
            source.mkdir()
            for name in ("config.json", "tokenizer.json"):
                (source / name).write_text("{}", encoding="utf-8")
            (source / "model.safetensors.index.json").write_text(
                '{"weight_map":{"x":"model.safetensors-00001-of-00001.safetensors"}}',
                encoding="utf-8",
            )
            shard = source / "model.safetensors-00001-of-00001.safetensors"
            shard.write_text("weights", encoding="utf-8")
            self.assertEqual(materialize_checkpoint_view(source, root / "view"), source)


if __name__ == "__main__":
    unittest.main()
