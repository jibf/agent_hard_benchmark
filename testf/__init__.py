import sys
import os
sys.path.append(os.path.abspath('../..'))

from Drafter_Bench.testf.test_types import (
    fileobject,
    TEXT_ALIGN_LEFT,
    TEXT_ALIGN_RIGHT,
    TEXT_ALIGN_JUSTIFY,
    TEXT_ALIGN_CENTER,
)
from Drafter_Bench.testf.functions import (
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
from Drafter_Bench.testf.metric import groundcheck, cross_check

