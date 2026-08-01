"""Guard against silently dropped Config -> AutoPath kwarg forwarding in the CLI.

autopath/cli/run_AutoPath.py constructs AutoPath(...) from a Config using a
hand-written, explicit kwarg list. If a parameter is added to both
Config.__init__ and AutoPath.__init__ but the CLI kwarg list isn't updated,
users who set that option in their JSON config get silently ignored (no
error, no warning) -- the feature is inert through the CLI entry point.

This test parses the CLI module with `ast` (rather than grepping raw text)
so it can't be fooled by the name appearing in a comment or docstring; it
must actually be a keyword argument in the AutoPath(...) call.
"""

import ast
from pathlib import Path

CLI_PATH = Path(__file__).resolve().parents[1] / "autopath" / "cli" / "run_AutoPath.py"

# Parameters that must be forwarded from Config to AutoPath in the CLI.
REQUIRED_FORWARDED_KWARGS = [
    "sMD_conv_window",
    "sMD_conv_streak",
    "sMD_autostop_estimator",
    "sMD_alternate_speeds",
]


def _find_autopath_call_kwargs():
    source = CLI_PATH.read_text()
    tree = ast.parse(source, filename=str(CLI_PATH))

    autopath_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AutoPath"
    ]

    assert len(autopath_calls) == 1, (
        f"Expected exactly one AutoPath(...) call in {CLI_PATH}, "
        f"found {len(autopath_calls)}"
    )

    call = autopath_calls[0]
    return {kw.arg for kw in call.keywords if kw.arg is not None}


def test_autostop_kwargs_forwarded_to_autopath():
    kwarg_names = _find_autopath_call_kwargs()

    missing = [name for name in REQUIRED_FORWARDED_KWARGS if name not in kwarg_names]

    assert not missing, (
        "AutoPath(...) call in run_AutoPath.py is missing forwarded kwargs "
        f"for: {missing}. These exist on both Config and AutoPath but are "
        "not passed through the CLI, so setting them in a JSON config is "
        "silently ignored."
    )
