import asyncio
import logging


async def safe_llm_invoke(llm, messages, max_retries=5):
    base_delay = 2

    for attempt in range(max_retries):
        if attempt > 0:
            await asyncio.sleep(0.5)
        try:
            return await llm.ainvoke(messages)

        except Exception as e:
            error_str = str(e).lower()

            if "429" in error_str or "rate limit" in error_str:
                wait_time = base_delay * (2 ** attempt)
                logging.warning(
                    "[RATE LIMIT] Retry %s in %ss...",
                    attempt + 1,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                raise e

    raise Exception("Max retries exceeded due to rate limits")
