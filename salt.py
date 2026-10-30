import hashlib, os


def get_student_params(student_id: str):
    h = int(hashlib.sha256(student_id.encode('utf-8')).hexdigest()[:8], 16)
    # Salted matrix dimension N between 1024 and 1536 (multiple of 64)
    N = 1024 + ((h % 9) * 64)
    # Salted stride benchmark buffer size: 2^24 elements (~134 MB double-precision)
    buffer_elements = 16_777_216
    # Seed for deterministic random number generation
    seed = h % 100_007
    return {"student_id": student_id, "N": N, "buffer_elements": buffer_elements, "seed": seed}


def load_params(results_dir=None):
    """Read results/salted_params.json produced by profile_env.py."""
    import json
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    with open(os.path.join(results_dir, "salted_params.json")) as f:
        return json.load(f)
