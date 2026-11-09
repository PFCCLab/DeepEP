# SPDX-License-Identifier: MIT
import ast
import re
import unittest
from pathlib import Path


SETUP_PATH = Path(__file__).resolve().parents[1] / "setup.py"


def get_hardcoded_cxx_standard_flags(function_name: str):
    tree = ast.parse(SETUP_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"-std=c\+\+\d+", node.value)
    ]


class TestSetupCompileFlags(unittest.TestCase):
    def test_hybrid_ep_inherits_torch_cxx_standard(self):
        self.assertEqual(
            get_hardcoded_cxx_standard_flags("get_extension_hybrid_ep_cpp"),
            [],
            "hybrid_ep_cpp must inherit the C++ standard selected by PyTorch",
        )


if __name__ == "__main__":
    unittest.main()
