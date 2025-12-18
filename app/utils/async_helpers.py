
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any



_thread_pool = ThreadPoolExecutor(max_workers=4)


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:


    if sys.version_info >= (3, 9):
        return await asyncio.to_thread(func, *args, **kwargs)
    else:

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_thread_pool, lambda: func(*args, **kwargs))