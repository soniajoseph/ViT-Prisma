@dataclass
class LensHandle:
    """Dataclass that holds information about a PyTorch hook."""

    hook: hooks.RemovableHandle
    """Reference to the Hook's Removable Handle."""

    is_permanent: bool = False
    """Indicates if the Hook is Permanent."""

    context_level: Optional[int] = None
    """Context level associated with the hooks context manager for the given hook."""


