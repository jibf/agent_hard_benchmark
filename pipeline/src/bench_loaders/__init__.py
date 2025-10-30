from .base_loader import BaseLoader
from .tau_bench_loader import TauBenchLoader
from .tau2_bench_loader import Tau2BenchLoader
from .ace_bench_loader import AceBenchLoader
from .nexus_bench_loader import NexusBenchLoader
# from .tool_sandbox_loader import ToolSandBoxLoader
from .complex_func_bench_loader import ComplexFuncBenchLoader
from .drafter_bench_loader import DrafterBenchLoader
from .bfcl_loader import BfclLoader
from .bfcl_v4_loader import BfclV4Loader
from .multi_challenge_loader import MultiChallengeLoader

from ..utils.types import Benchmark

def get_bench_loader(benchmark: Benchmark):
    """Get the appropriate loader class for the given benchmark."""
    loader_map = {
        Benchmark.TAU_BENCH: TauBenchLoader,
        Benchmark.TAU2_BENCH: Tau2BenchLoader,
        Benchmark.COMPLEX_FUNC_BENCH: ComplexFuncBenchLoader,
        Benchmark.NEXUS_BENCH: NexusBenchLoader,
        Benchmark.DRAFTER_BENCH: DrafterBenchLoader,
        Benchmark.ACE_BENCH: AceBenchLoader,
        Benchmark.BFCL: BfclLoader,
        Benchmark.BFCL_V4: BfclV4Loader,
        Benchmark.MULTI_CHALLENGE: MultiChallengeLoader,
        # Benchmark.TOOL_SANDBOX: ToolSandBoxLoader
    }
    
    if benchmark not in loader_map:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    return loader_map[benchmark]
