import time, sys
from multiprocessing import get_context, cpu_count

# Make it global variables so that worker processes can access them quickly
_G = _eps = _mu = _gamma = _sim = None

def _init_pool(G, eps, mu, gamma, similarity_func):
    global _G, _eps, _mu, _gamma, _sim
    _G, _eps, _mu, _gamma, _sim = G, eps, mu, gamma, similarity_func

def _is_eps_neighbor(u, v):
    return _sim(_G, u, v, _gamma) >= _eps

def _core_of_u(u):
    cnt = 0
    for v in _G.neighbors(u):
        if _is_eps_neighbor(u, v):
            cnt += 1
            if cnt >= _mu:
                # No need to count more: core confirmed
                return (u, True)
    return (u, False)

def run_parallel_cores(G, similarity_func, eps=0.5, mu=2, gamma=1, workers=None, chunksize=32):
    """Determine core nodes in parallel and return as a set"""
    if workers is None:
        workers = max(1, cpu_count() - 1)

    # Linux/mac : fork, Windows : spawn
    start_method = "fork" if sys.platform != "win32" else "spawn"
    ctx = get_context(start_method)

    n_start = time.time()
    with ctx.Pool(processes=workers,
                  initializer=_init_pool,
                  initargs=(G, eps, mu, gamma, similarity_func)) as pool:
        cores = {u for (u, is_core) in pool.imap_unordered(_core_of_u, G.nodes(), chunksize=chunksize) if is_core}

    similarity_calculating_time = time.time() - n_start
    return cores, similarity_calculating_time


def _core_of_u_include_me(u):
    cnt = 1
    for v in _G.neighbors(u):
        if _is_eps_neighbor(u, v):
            cnt += 1
            if cnt >= _mu:
                # No need to count more: core confirmed
                return (u, True)
    return (u, False)

def run_parallel_cores_include_me(G, similarity_func, eps=0.5, mu=2, gamma=1, workers=None, chunksize=32):
    """Determine core nodes in parallel and return as a set"""
    if workers is None:
        workers = max(1, cpu_count() - 1)

    # Linux/mac : fork, Windows : spawn
    start_method = "fork" if sys.platform != "win32" else "spawn"
    ctx = get_context(start_method)

    n_start = time.time()
    with ctx.Pool(processes=workers,
                  initializer=_init_pool,
                  initargs=(G, eps, mu, gamma, similarity_func)) as pool:
        cores = {u for (u, is_core) in pool.imap_unordered(_core_of_u_include_me, G.nodes(), chunksize=chunksize) if is_core}

    similarity_calculating_time = time.time() - n_start
    return cores, similarity_calculating_time