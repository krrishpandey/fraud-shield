"""The call script (brief section 5), used to write SIMULATED transcripts. English only.

The real agent's prompt lives in fraudshield/voice/agent_config.json; these lines mirror it so a simulated call reads
like the real one. Every simulated transcript is labelled simulated in the UI and the audit log.
"""
from __future__ import annotations

from typing import Any

LINES: dict[str, str] = {
    "open": "Hello, this is Tracd calling for {company_name} about your courier account. We will never ask for "
            "your OTP, password or card number. A booking was just made on your account: {weight_kg} kg "
            "{service}, {origin} to {destination}, at {booking_time}{device_note}. Was this you?",
    "yes": "Yes, that was us.",
    "no": "No, that was not me. We did not book anything today.",
    "unsure": "I'm not sure. Let me check with my staff.",
    "deny_close": "Thank you. We have stopped this booking and secured your account. Our team will contact you "
                  "shortly.",
    "ask_detail": "Thanks. Which pickup address is this going from?",
    "ask_change": "Has anything changed, like a new warehouse or a new staff login?",
    "no_change": "No, nothing has changed.",
    "confirm_close": "Thank you, I have noted that. Your booking will be updated in a minute. Goodbye.",
    "unsure_close": "No problem. A member of our team will follow up with you shortly. The booking stays on "
                    "hold until then.",
}
DEVICE_NOTE = ", from a new device"


def simulated_transcript(language: str, variables: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    """Bolna-style 'assistant: / user:' lines for a simulated owner answer (yes / no / unsure)."""
    L = LINES
    v = {**variables, "device_note": DEVICE_NOTE if variables.get("device_is_new") == "yes" else ""}
    out = ["assistant: " + L["open"].format(**v)]
    said = answer.get("owner_confirmed")
    if said == "no":
        out += ["user: " + (answer.get("owner_line") or L["no"]), "assistant: " + L["deny_close"]]
    elif said == "yes":
        out += ["user: " + (answer.get("owner_line") or L["yes"]), "assistant: " + L["ask_detail"],
                "user: " + str(answer.get("detail_answer") or ""), "assistant: " + L["ask_change"],
                "user: " + str(answer.get("change_line") or answer.get("change_reason") or L["no_change"]),
                "assistant: " + L["confirm_close"]]
    else:
        out += ["user: " + (answer.get("owner_line") or L["unsure"]), "assistant: " + L["unsure_close"]]
    return out
