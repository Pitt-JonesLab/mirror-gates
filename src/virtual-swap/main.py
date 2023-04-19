"""Sample docstring."""


class Electron:
    """A representation of an electron."""

    def __init__(self, size: str = None, name: str = None) -> None:
        """Create an electron.

        :param size: How big should this thing be?
        :param name: The name we'll call the electron. Nicknames preferred.
        :raises ValueError: You did something wrong
        """
        self.size = size
        self.name = name

    @property
    def charge(self):
        """The charge of the electron."""
        return -1

    @property
    def mass(self) -> float:
        """The mass of the electron."""
        return 2.1

    def compute_momentum(self, velocity: float) -> float:
        """Compute the electron's velocity.

        :param velocity: The electron's velocity
        :return: The computed momentum.
        :raises ValueError: You did something wrong
        """
        return self.mass * velocity
