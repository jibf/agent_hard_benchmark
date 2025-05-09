import sys
import os

sys.path.append(os.path.abspath('../..'))

from DrafterBench.utils.testf.test_types import (
    fileobject,
    TEXT_ALIGN_LEFT,
    TEXT_ALIGN_RIGHT,
    TEXT_ALIGN_JUSTIFY,
    TEXT_ALIGN_CENTER,
)
from DrafterBench.utils.testf.functions import (
    open,
    extractanno,
    selector,
    select_from_drawings,
    manipulate_text,
    extract_table,
    manipulate_table,
    draw_drawer,
    delete,
    repairer,
    manipulate_draw,
    recorder,
    Projector,
    project_draw,
)
