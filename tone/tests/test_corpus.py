from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tone  # noqa: E402

KB_DIR = ROOT / "knowledge-base"


class CorpusTest(unittest.TestCase):
    """Язык документов базы знаний не должен давать жёстких срабатываний."""

    def test_every_document_is_clean(self) -> None:
        if not KB_DIR.is_dir():
            self.skipTest("knowledge-base отсутствует")
        documents = sorted(KB_DIR.glob("*.txt"))
        self.assertGreaterEqual(len(documents), 10)

        for path in documents:
            with self.subTest(document=path.name):
                text = path.read_text(encoding="utf-8-sig")
                hard = [v for v in tone.check(text) if v.severity == tone.HARD]
                self.assertEqual(hard, [], f"{path.name}: {[(v.code, v.quote) for v in hard[:3]]}")

    def test_normalize_keeps_document_text(self) -> None:
        if not KB_DIR.is_dir():
            self.skipTest("knowledge-base отсутствует")
        sample = KB_DIR / "законы__ТК_ст70_действующая.txt"
        if not sample.is_file():
            self.skipTest("нет образца")
        text = sample.read_text(encoding="utf-8-sig")
        normalized = tone.normalize(text)
        self.assertEqual(normalized.text, " ".join(text.split()))


if __name__ == "__main__":
    unittest.main()
