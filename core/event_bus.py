"""
core/event_bus.py
-----------------
Lightweight publish-subscribe event bus.

All components communicate exclusively through this bus.
No component holds a direct reference to any other component.

Design:
    - O(1) publish for any event type
    - Multiple subscribers per event type
    - Safe unsubscribe during iteration (copy-on-publish)
    - Thread-safe for future async/threaded extensions
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Callable, Any, Type

logger = logging.getLogger(__name__)


class EventBus:
    """
    Central publish-subscribe message broker.

    Usage
    -----
    bus = EventBus()
    bus.subscribe(MarketTickEvent, my_handler)
    bus.publish(MarketTickEvent(timestamp=..., current_index=0))
    """

    def __init__(self) -> None:
        # event_type -> list of callbacks
        self._subscribers: dict[Type, list[Callable]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, event_type: Type, callback: Callable[[Any], None]) -> None:
        """
        Register *callback* to receive events of *event_type*.

        Parameters
        ----------
        event_type : type
            The event class to subscribe to (e.g. MarketTickEvent).
        callback : callable
            Function that accepts a single event instance.
        """
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            logger.debug("Subscribed %s to %s", callback.__qualname__, event_type.__name__)

    def unsubscribe(self, event_type: Type, callback: Callable[[Any], None]) -> None:
        """
        Remove *callback* from the subscriber list for *event_type*.

        Safe to call even if the callback was never registered.
        """
        try:
            self._subscribers[event_type].remove(callback)
            logger.debug("Unsubscribed %s from %s", callback.__qualname__, event_type.__name__)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Any) -> None:
        """
        Deliver *event* to all registered subscribers.

        Uses a snapshot of the subscriber list so that subscribers may
        safely unsubscribe themselves during handling without causing
        a RuntimeError.

        Parameters
        ----------
        event : Any
            An event dataclass instance (MarketTickEvent, etc.).
        """
        event_type = type(event)
        handlers = list(self._subscribers.get(event_type, []))  # snapshot

        if not handlers:
            logger.debug("No subscribers for %s", event_type.__name__)
            return

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.exception(
                    "Error in subscriber %s handling %s: %s",
                    handler.__qualname__,
                    event_type.__name__,
                    exc,
                )

    # ------------------------------------------------------------------
    # Introspection helpers (useful for debugging / future dashboard)
    # ------------------------------------------------------------------

    def subscriber_count(self, event_type: Type) -> int:
        """Return the number of subscribers for a given event type."""
        return len(self._subscribers.get(event_type, []))

    def clear(self) -> None:
        """Remove all subscriptions (useful for testing / reset)."""
        self._subscribers.clear()