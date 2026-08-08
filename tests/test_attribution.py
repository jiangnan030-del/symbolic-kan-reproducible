from pathlib import Path

UPSTREAM = "https://github.com/sfaroughi3/Pub_Symbolic_KANs"
COMMIT = "9481a822e73e5a7520c6c0a425a8a402f2878c03"


def test_provenance_is_prominent_and_complete() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    notice = (root / "NOTICE.md").read_text(encoding="utf-8")
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    for text in (readme, notice):
        assert UPSTREAM in text
        assert COMMIT in text
        assert "unofficial" in text.lower()
    assert "Copyright (c) 2024 Prof. Salah A. Faroughi" in license_text
    assert "MIT License" in license_text
