"""Package-level invariants: the public surface is what it claims to be."""

from __future__ import annotations

import hero


def test_all_names_are_importable():
    """Every name in __all__ actually resolves, so the surface cannot rot."""
    missing = [name for name in hero.__all__ if not hasattr(hero, name)]
    assert not missing, f"__all__ lists unimportable names: {missing}"


def test_all_is_sorted_and_unique():
    assert hero.__all__ == sorted(set(hero.__all__))


def test_public_functions_are_exported():
    """Guards against adding a public helper and forgetting to export it."""
    from hero import core, rewards

    expected = {
        name
        for module in (core, rewards)
        for name in vars(module)
        if not name.startswith("_")
        and callable(getattr(module, name))
        and getattr(getattr(module, name), "__module__", "").startswith("hero.")
    }
    assert expected <= set(hero.__all__), sorted(expected - set(hero.__all__))
