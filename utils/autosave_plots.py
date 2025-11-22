import os
import matplotlib.pyplot as plt
from datetime import datetime

# Path to central folder
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports/figures")

# Create folder if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Backup original plt.show()
_original_show = plt.show

def _autosave_show(*args, **kwargs):
    # Generate timestamp-based filename
    filename = datetime.now().strftime("plot_%Y%m%d_%H%M%S_%f.png")
    filepath = os.path.join(SAVE_DIR, filename)

    # Save current figure
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    # Call original show()
    _original_show(*args, **kwargs)

# Monkey-patch
plt.show = _autosave_show
