"""The verifier's band floor must hold in code, not just in the prompt.

The system prompt asks Claude to stay inside its action band. A prompt is a request.
The one failure that actually costs money is the model talking us DOWN from a block --
so that direction is enforced here and asserted below.

Run: python backend/test_verify.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ml"))
from main import clamp_action  # noqa: E402


def main():
    # escalating caution is allowed -- the verifier may see something the model didn't
    assert clamp_action("auto_block", "auto_clear") == ("auto_block", False)
    assert clamp_action("hold_for_review", "auto_clear") == ("hold_for_review", False)
    assert clamp_action("auto_block", "hold_for_review") == ("auto_block", False)

    # de-escalating below the band floor is overridden, and flagged
    assert clamp_action("auto_clear", "auto_block") == ("auto_block", True)
    assert clamp_action("hold_for_review", "auto_block") == ("auto_block", True)
    assert clamp_action("auto_clear", "hold_for_review") == ("hold_for_review", True)

    # staying put is a no-op
    for band in ("auto_block", "hold_for_review", "auto_clear"):
        assert clamp_action(band, band) == (band, False)

    # anything unrecognised (hallucinated action, null, wrong type) falls back to the band
    for junk in ("escalate", "APPROVE", "", None, 0):
        assert clamp_action(junk, "hold_for_review") == ("hold_for_review", True), junk

    print("band floor holds: the verifier cannot de-escalate below the model's score")


if __name__ == "__main__":
    main()
