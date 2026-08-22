"""Celery-backed background job entrypoints.

Handler contract (breaking change; previously ``(Session, BackgroundJob)``):
a handler now takes one positional ``BackgroundJobRef`` -- an immutable value
snapshot of the row -- and returns the result dict. No Session is handed in,
because the worker no longer keeps one open across the handler body; a handler
that needs the database opens its own short session per operation
(``xagent.web.models.database.session_scope``).

Migrating a downstream handler::

    def handle(db, job):                  # before
        user = db.get(User, job.user_id)
        ...

    def handle(ref: BackgroundJobRef):    # after
        with session_scope() as db:
            user = db.get(User, ref.user_id)
        ...

``register_background_job_handler`` rejects the old two-argument shape at
registration time rather than letting it fail once per attempt.
"""

from ..models.background_job import BackgroundJobRef
from .tasks import (
    BackgroundJobHandler,
    is_background_job_handler_registered,
    register_background_job_handler,
)

__all__ = [
    "BackgroundJobHandler",
    "BackgroundJobRef",
    "is_background_job_handler_registered",
    "register_background_job_handler",
]
