"""Read the previous study schema and normalize obsolete terminology in memory."""


def normalize_study(payload):
    """Convert legacy keys only when reading a pre-revision result."""
    if "preliminary_noise_floor" in payload:
        payload.setdefault("absolute_inter_seed_spread",payload.pop("preliminary_noise_floor"))
    for entry in payload.get("divergence_summary",{}).values():
        if "noise_floor_seed_to_seed" in entry:
            entry.setdefault("absolute_inter_seed_spread",entry.pop("noise_floor_seed_to_seed"))
    return payload
