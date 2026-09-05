"""1D sweep axis → config field mapping (no torch imports)."""

# CLI sweep axis -> system_model Pydantic field (snr is lowercase; N/M/T are uppercase)
SWEEP_AXIS_TO_SYSTEM_MODEL_FIELD = {
    "snr": "snr",
    "n": "N",
    "m": "M",
    "t": "T",
}


def system_model_override_for_axis(axis: str, value) -> str:
    """Build a dot-path override for a 1D sweep axis."""
    field = SWEEP_AXIS_TO_SYSTEM_MODEL_FIELD.get(axis.lower())
    if field is None:
        raise ValueError(f"Unsupported sweep axis for system_model override: {axis}")
    return f"system_model.{field}={value}"
