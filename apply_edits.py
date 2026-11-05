"""Apply a saved weight edit (compact delta: only the changed rows / columns / entries) to a loaded model.

    from chronos import Chronos2Pipeline
    from apply_edits import apply_edit
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
    apply_edit(pipe.model, "edits/chronos2_nanfix_agent.npz")   # the agent's 1,536-weight fix

Delta format (npz): for a tensor named NAME, either NAME|rows + NAME|row_values, NAME|cols + NAME|col_values,
or NAME|idx + NAME|values (1-D). Values are the NEW values of those entries, not differences.
"""

import numpy as np
import torch


def apply_edit(model, path):
    sd = model.state_dict()
    z = np.load(path)
    names = {k.split("|")[0] for k in z.files}
    with torch.no_grad():
        for name in names:
            t = sd[name]
            if f"{name}|rows" in z.files:
                t[torch.as_tensor(z[f"{name}|rows"])] = torch.as_tensor(z[f"{name}|row_values"], dtype=t.dtype)
            elif f"{name}|cols" in z.files:
                t[:, torch.as_tensor(z[f"{name}|cols"])] = torch.as_tensor(z[f"{name}|col_values"], dtype=t.dtype)
            else:
                t[torch.as_tensor(z[f"{name}|idx"])] = torch.as_tensor(z[f"{name}|values"], dtype=t.dtype)
    return model


def to_full_npz(model, path, out):
    """Write the edited tensors in full (the format c2_official.py --edit and dev_eval.py --edit read)."""
    apply_edit(model, path)
    names = {k.split("|")[0] for k in np.load(path).files}
    np.savez(out, **{n: model.state_dict()[n].detach().cpu().numpy() for n in names})
