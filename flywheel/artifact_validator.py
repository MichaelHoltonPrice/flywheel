"""Project-supplied artifact validation.

Flywheel finalizes artifacts at multiple sites: ``flywheel
import artifact`` and post-block output collection (one-shot
container, request-response container, agent block).  Every
site consults the same per-project
:class:`ArtifactValidatorRegistry` so the on-disk shape of any
committed artifact is whatever the project says is valid,
regardless of how the artifact got there.

The validator's contract is uniform across sites: **the
validator is given a directory containing the bytes that would
become the artifact instance, and decides yes or no.**  The
validator does not need to know whether those bytes came from
a container's output tempdir, a staged copy of an operator's
import, or anywhere else — that is flywheel's concern, not the
validator's.

Validators are plain Python callables.  Each maps to one
artifact declaration name and is invoked exactly once per
candidate artifact, *before* the candidate becomes a permanent
:class:`flywheel.artifact.ArtifactInstance`.

Failure semantics depend on the site:

* For ``flywheel import artifact``, a rejected candidate is
  discarded and the workspace state is untouched.
* For block-produced outputs, a rejected slot is dropped while
  any sibling slots that pass validation are still committed.
  The execution record is marked ``failed`` with
  :data:`flywheel.runtime.FAILURE_OUTPUT_VALIDATE` and the
  per-slot rejection reasons appear on
  :attr:`flywheel.artifact.BlockExecution.error`.

Validators receive the artifact declaration so they can branch
on declared kind / git fields if needed.  They must treat the
candidate directory as read-only.

Registries are constructed by the project (typically by a
factory referenced from ``flywheel.yaml`` under the
``artifact_validators`` key) and resolved by
:meth:`flywheel.config.ProjectConfig.load_artifact_validator_registry`.
A workspace with no registered validators is fully supported —
the registry is consulted but every name is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flywheel.template import ArtifactDeclaration


class ArtifactValidationError(Exception):
    """Raised when a candidate artifact fails its declared validator.

    Attributes:
        name: The artifact declaration name whose validator
            rejected the candidate.  ``None`` only when raised
            directly (not via a registry lookup) — e.g. by
            tests.
        path: The candidate path that was rejected.  ``None``
            in the same edge case.

    The string form of the exception is the message validators
    should pass — keep it short and human-readable; it surfaces
    on :attr:`flywheel.artifact.BlockExecution.error` for failed
    block outputs and is printed verbatim by ``flywheel import
    artifact``.
    """

    def __init__(
        self,
        message: str,
        *,
        name: str | None = None,
        path: Path | None = None,
    ) -> None:
        """Build a validation error.

        ``message`` becomes the exception's ``str()`` form;
        ``name`` and ``path`` are optional structured fields
        attached for callers that want them (the registry
        backfills both when the validator omits them).
        """
        super().__init__(message)
        self.name = name
        self.path = path


ArtifactValidator = Callable[
    [str, "ArtifactDeclaration", Path],
    None,
]
"""Project-supplied callable.  Inspects a candidate artifact at
``staged_path``; raises :class:`ArtifactValidationError` to reject
it; returns ``None`` to accept it.

Signature: ``validator(name, declaration, staged_path) -> None``.
``staged_path`` is a directory containing the bytes flywheel will
commit as the artifact instance if the validator accepts it.  How
those bytes got there — a container output tempdir, a staged copy
of an operator's import — is flywheel's concern, not the
validator's.  The callable must treat ``staged_path`` as
read-only."""


class ArtifactValidatorRegistry:
    """Project-owned map from artifact declaration name to validator.

    A registry is consulted at every artifact-finalization site.
    Names with no registered validator are accepted unconditionally;
    flywheel never invents a default validator.

    Project hooks construct one registry instance per process,
    typically in a factory referenced from ``flywheel.yaml``::

        # flywheel.yaml
        artifact_validators: myproject.validators:build_registry

        # myproject/validators.py
        def build_registry() -> ArtifactValidatorRegistry:
            r = ArtifactValidatorRegistry()
            r.register("checkpoint", check_checkpoint_pt)
            return r

    The registry is **immutable from flywheel's perspective**:
    flywheel only reads it.  Projects that want to register
    validators dynamically (per-workspace, per-environment, etc.)
    do so before flywheel ever sees the registry.
    """

    def __init__(
        self,
        validators: Mapping[str, ArtifactValidator] | None = None,
    ) -> None:
        """Seed the registry from an optional initial mapping."""
        self._validators: dict[str, ArtifactValidator] = (
            dict(validators) if validators else {}
        )

    def register(
        self, name: str, validator: ArtifactValidator,
    ) -> None:
        """Register a validator for an artifact declaration name.

        Replaces any previously-registered validator for the
        same name without warning; "last write wins" by design,
        so a project hook can override a battery's default.
        """
        self._validators[name] = validator

    def has(self, name: str) -> bool:
        """Return ``True`` iff a validator is registered for ``name``."""
        return name in self._validators

    def names(self) -> list[str]:
        """Return the sorted list of registered artifact names.

        Useful for tests and CLI introspection.  The internal
        dict is never exposed to mutation.
        """
        return sorted(self._validators)

    def validate(
        self,
        name: str,
        declaration: ArtifactDeclaration | None,
        staged_path: Path,
    ) -> None:
        """Invoke the registered validator for ``name``, if any.

        No-op when no validator is registered (an unvalidated
        artifact is treated as valid by definition — the project
        opted out of declaring rules for it).

        Args:
            name: The artifact declaration name.
            declaration: The full declaration the project
                registered the validator against.  May be
                ``None`` for callers that do not have one
                handy (the validator must tolerate this if it
                opts in to it).
            staged_path: Path to the candidate artifact on
                disk.  For copy artifacts this is a directory
                of files.

        Raises:
            ArtifactValidationError: When the validator rejects
                the candidate.  Other exception types are
                wrapped in :class:`ArtifactValidationError`
                with the original exception attached as
                ``__cause__`` — validators that raise plain
                exceptions are still reported as validation
                failures, never silent.
        """
        validator = self._validators.get(name)
        if validator is None:
            return
        try:
            validator(name, declaration, staged_path)  # type: ignore[arg-type]
        except ArtifactValidationError as exc:
            if exc.name is None:
                exc.name = name
            if exc.path is None:
                exc.path = staged_path
            raise
        except Exception as exc:  # noqa: BLE001
            raise ArtifactValidationError(
                f"validator for {name!r} raised "
                f"{type(exc).__name__}: {exc}",
                name=name,
                path=staged_path,
            ) from exc
