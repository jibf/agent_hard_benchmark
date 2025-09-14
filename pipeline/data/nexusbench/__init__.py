# Ensure that imports expecting the top-level package name `nexusbench` work
# even when the library is located under `data.nexusbench` inside this repo.
# We achieve this by aliasing the current module to `nexusbench` in
# `sys.modules` at import time.

import sys as _sys

# Alias the current module namespace (`data.nexusbench`) to `nexusbench`
# so that statements like `import nexusbench.tools` resolve correctly when
# executed from within the repository.
_sys.modules.setdefault("nexusbench", _sys.modules[__name__])

# Now expose the CLI entrypoint under both namespaces.
from .entrypoint import main  # noqa: E402