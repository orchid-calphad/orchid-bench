"""Unit tests for bench/common.py: canonical JSON, file helpers, hashing, number formatting and paths."""

from __future__ import annotations

import hashlib
import math

import common
import numpy as np
import pytest

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestCanonicalJson:
    """canonical_json serialises deterministically, so equal data always gives equal bytes."""

    def test_sorts_keys_and_uses_compact_separators(self):
        """Keys are sorted, separators carry no spaces, and the text ends with one newline."""
        assert common.canonical_json({"b": 1, "a": [1.5, 2]}) == '{"a":[1.5,2],"b":1}\n'

    def test_keeps_non_ascii_text_unescaped(self):
        """Non-ASCII text such as the degree sign is written as-is rather than escaped."""
        assert common.canonical_json({"unit": "°C"}) == '{"unit":"°C"}\n'

    def test_converts_numpy_scalars_and_tuples_to_plain_json(self):
        """NumPy scalars become plain numbers and tuples become lists."""
        text = common.canonical_json({"n": np.int64(3), "x": np.float64(0.25), "t": (1, 2)})
        assert text == '{"n":3,"t":[1,2],"x":0.25}\n'

    def test_stringifies_non_string_keys(self):
        """Integer keys are written as JSON string keys."""
        assert common.canonical_json({1: "a"}) == '{"1":"a"}\n'

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_floats(self, bad):
        """Non-finite floats raise instead of producing invalid JSON."""
        with pytest.raises(ValueError, match="non-finite"):
            common.canonical_json({"v": bad})


class TestWriteAndLoadJson:
    """write_json and load_json round-trip UTF-8 data and create the parent directories they need."""

    def test_round_trip_preserves_data(self, tmp_path):
        """A written document loads back equal to the object that was written."""
        path = tmp_path / "x.json"
        obj = {"name": "Ag–Cu", "T": [1000.0, 1234.5]}
        common.write_json(path, obj)
        assert common.load_json(path) == obj

    def test_creates_missing_parent_directories(self, tmp_path):
        """Parent directories are created on demand."""
        path = tmp_path / "a" / "b" / "c.json"
        common.write_json(path, {"k": 1})
        assert path.is_file()

    def test_compact_file_content_is_canonical_json(self, tmp_path):
        """The compact output, read back in text mode, is exactly canonical_json of the object."""
        path = tmp_path / "x.json"
        common.write_json(path, {"b": 2, "a": 1})
        assert path.read_text(encoding="utf-8") == common.canonical_json({"b": 2, "a": 1})

    def test_pretty_output_is_indented_with_sorted_keys(self, tmp_path):
        """Pretty output indents by one space, sorts keys and ends with a newline."""
        path = tmp_path / "p.json"
        common.write_json(path, {"b": 2, "a": 1}, pretty=True)
        assert path.read_text(encoding="utf-8") == '{\n "a": 1,\n "b": 2\n}\n'

    def test_non_ascii_text_is_stored_as_utf8(self, tmp_path):
        """Non-ASCII characters are stored as their UTF-8 bytes, not as escapes or a legacy codepage."""
        path = tmp_path / "u.json"
        common.write_json(path, {"s": "µ°"})
        content = path.read_bytes().replace(b"\r\n", b"\n")  # text mode writes CRLF on Windows
        assert content == '{"s":"µ°"}\n'.encode()


class TestFmt:
    """fmt prints exactly the requested number of decimals and never prints a negative zero."""

    @pytest.mark.parametrize(
        ("value", "nd", "expected"),
        [
            (1.23456, 3, "1.235"),
            (2, 2, "2.00"),
            (-1.5, 1, "-1.5"),
            (1e6, 3, "1000000.000"),
        ],
    )
    def test_prints_requested_decimals(self, value, nd, expected):
        """The output has exactly nd decimals and keeps the sign of non-zero values."""
        assert common.fmt(value, nd) == expected

    @pytest.mark.parametrize(("value", "nd"), [(-0.0001, 2), (-0.0, 1)])
    def test_negative_zero_loses_its_sign(self, value, nd):
        """A value that rounds to zero is printed without a minus sign."""
        assert common.fmt(value, nd).startswith("0.")
        assert not common.fmt(value, nd).startswith("-")


class TestRound:
    """r rounds to a fixed number of decimal places and returns a float."""

    def test_rounds_to_requested_places(self):
        """1.23456 rounded to two places is 1.23, returned as a float."""
        value = common.r(1.23456, 2)
        assert isinstance(value, float)
        assert value == pytest.approx(1.23)

    def test_accepts_numeric_strings(self):
        """A numeric string is converted before rounding."""
        assert common.r("3.14159", 3) == pytest.approx(3.142)


class TestEl:
    """el turns an upper-case element symbol into conventional capitalisation."""

    @pytest.mark.parametrize(("symbol", "expected"), [("AL", "Al"), ("cu", "Cu"), ("NI", "Ni"), ("Ag", "Ag"), ("a", "A")])
    def test_capitalises_first_letter_only(self, symbol, expected):
        """The first letter is upper case and the rest lower case."""
        assert common.el(symbol) == expected


class TestSha256File:
    """sha256_file streams a file and returns the same digest as hashlib over its bytes."""

    def test_empty_file_has_the_empty_digest(self, tmp_path):
        """An empty file hashes to the SHA-256 of no bytes."""
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert common.sha256_file(path) == EMPTY_SHA256

    def test_multi_chunk_file_matches_hashlib(self, tmp_path):
        """A file larger than the 1 MiB read chunk hashes to the same digest as a single pass."""
        data = np.random.default_rng(0).bytes((1 << 20) + 12345)
        path = tmp_path / "big.bin"
        path.write_bytes(data)
        assert common.sha256_file(path) == hashlib.sha256(data).hexdigest()


class TestPaths:
    """The path constants point into the repository checkout that the tests run against."""

    def test_directories_sit_under_the_bench_folder(self, bench_dir, repo_dir):
        """bench, the repository root and the data folders resolve to their expected locations."""
        assert common.bench == bench_dir
        assert common.root == repo_dir
        assert common.docs_dir == bench_dir / "docs"
        assert common.truth_dir == bench_dir / "truth"
        assert common.cache == bench_dir / "cache"
        assert common.results == bench_dir / "results"

    def test_physical_constants_are_si(self):
        """The gas constant and standard pressure carry the values the code is written against."""
        assert common.r_gas == pytest.approx(8.314462618)
        assert common.p_atm == pytest.approx(101325.0)
