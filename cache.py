
import time
import hashlib

CACHE = {}

CACHE_TTL = 600  # 10 minutes


def get_cache_key(question):
    return hashlib.sha256(
        question.strip().lower().encode()
    ).hexdigest()


def get_cached_answer(question):

    key = get_cache_key(question)

    item = CACHE.get(key)

    if item is None:
        return None

    answer, created_at = item

    if time.time() - created_at > CACHE_TTL:
        del CACHE[key]
        return None

    return answer


def save_cached_answer(question, answer):

    key = get_cache_key(question)

    CACHE[key] = (answer, time.time())
