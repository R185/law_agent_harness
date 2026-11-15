from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE = ROOT / "tone"


class BoundaryTest(unittest.TestCase):
    """Слой не зависит от пайплайна и не тянет за собой чужие файлы."""

    def test_no_rag_dependency(self) -> None:
        for path in sorted(PACKAGE.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            with self.subTest(module=path.name):
                self.assertIsNone(re.search(r"^\s*(?:from|import)\s+rag\b", source, re.M))
                self.assertNotIn("tone_integration", source)

    def test_no_network_client(self) -> None:
        for path in sorted(PACKAGE.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            with self.subTest(module=path.name):
                for forbidden in ["httpx", "requests", "urllib", "openai"]:
                    self.assertNotIn(f"import {forbidden}", source)
                    self.assertNotIn(f"from {forbidden}", source)

    def test_layer_does_not_rewrite_or_fallback(self) -> None:
        """Слой не вызывает модель и не владеет текстом отказа: только build/normalize/check."""
        import tone

        for name in ["enforce", "rewrite", "fallback", "FALLBACK"]:
            self.assertNotIn(name, dir(tone))

    def test_removed_artifacts_have_no_side_files(self) -> None:
        for stale in ["tone_check.py", "tone_integration.patch"]:
            self.assertFalse((ROOT / stale).exists(), stale)

    def test_root_readme_is_untouched(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").is_file() else ""
        self.assertNotIn("tone", readme.lower())

    def test_docs_live_in_tone_md(self) -> None:
        doc = ROOT / "TONE.md"
        self.assertTrue(doc.is_file())
        text = doc.read_text(encoding="utf-8")
        for marker in ["build_system_prompt", "normalize", "check", "оркестратор"]:
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
