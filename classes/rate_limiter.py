import asyncio
import time


class RateLimiter:
    """
    Implements a token bucket algorithm for rate limiting.
    """
    def __init__(self, rate: float, capacity: float):
        self.tokens = capacity
        self.capacity = capacity
        self.rate = rate
        self.last_updated = self._current_time()
        self.lock = asyncio.Lock()

    def _current_time(self):
        return time.monotonic()

    async def refill(self):
        """
        Adds tokens to the bucket based on the elapsed time.
        """
        now = self._current_time()
        elapsed = now - self.last_updated
        self.last_updated = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

    async def acquire(self, tokens: float = 1):
        """
        Acquires tokens from the bucket, waiting if necessary.
        """
        async with self.lock:
            while True:
                await self.refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                await asyncio.sleep(0.1)  # Yield to the event loop
