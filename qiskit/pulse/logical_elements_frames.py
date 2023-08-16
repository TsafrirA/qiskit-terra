# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
.. _pulse-logical-elements-frames:

=======================================
Logical Elements & Frames (:mod:`qiskit.pulse.logical_elements_frames`)
=======================================

"""
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from qiskit.pulse.exceptions import PulseError


class LogicalElement(metaclass=ABCMeta):
    """Base class of logical elements.

    """

    def __init__(self, index):
        """Create ``LogicalElement``.

        Args:
            index: The index of the logical element.
        """
        self._validate_index(index)
        self._index = index
        self._hash = hash(self.name)

    @property
    def index(self):
        """Return the ``index`` of this logical element."""
        return self._index

    @abstractmethod
    def _validate_index(self, index) -> None:
        """Raise a PulseError if the logical element ``index`` is invalid.

        Raises:
            PulseError: If ``index`` is not valid.
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return the name of this logical element."""
        pass

    def __repr__(self):
        return self.name

    def __eq__(self, other: "LogicalElement") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same ``index``.

        Args:
            other: The logical element to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._index == other._index

    def __hash__(self):
        return self._hash


class Qubit(LogicalElement):
    """Qubit logical element."""
    def __init__(self, index: int):
        """Qubit logical element.

        Args:
            index: Qubit index.
        """
        super().__init__(index)

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")

    @property
    def name(self):
        """Return the name of this qubit"""
        return f"Q{self.index}"


class Coupler(LogicalElement):
    """Coupler logical element."""
    def __init__(self, qubit_index_1: int, qubit_index_2: int):
        """Coupler logical element.

        The coupler ``index`` is defined as the ``tuple`` (``qubit_index_1``,``qubit_index_2``).

        Args:
            qubit_index_1: Index of the first qubit coupled by the coupler.
            qubit_index_2: Index of the second qubit coupled by the coupler.
        """
        super().__init__((qubit_index_1, qubit_index_2))

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the coupler ``index`` is invalid. Namely, check if coupled qubit indices are
        non-negative integers.

        Raises:
            PulseError: If ``index`` is invalid.
        """
        for qubit_index in index:
            if not isinstance(qubit_index, (int, np.integer)) or qubit_index < 0:
                raise PulseError("Both indices of coupled qubits must be non-negative integers")

    @property
    def name(self):
        """Return the name of this coupler"""
        return f"Coupler{self.index}"


class Frame:
    """Pulse module Frame."""
    def __init__(self, name: str):
        """Create ``Frame``.

        Args:
            name: A unique identifier used to identify the frame.
        """
        self._name = name
        self._hash = hash(name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the frame."""
        pass

    def __repr__(self):
        return self.name

    def __eq__(self, other: "Frame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type and name.

        Args:
            other: The frame to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._name == other._name

    def __hash__(self):
        return self._hash


class GenericFrame(Frame):
    """Pulse module GenericFrame.
    """
    def __init__(self, name: str, frequency: float, phase: float = None):
        """Create ``Frame``.

        Args:
            name: A unique identifier used to identify the frame.
            frequency: The initial frequency for the frame.
            phase (Optional): The initial phase for the frame. Defaults to zero.
        """
        self._name = name
        self._frequency = frequency
        self._phase = phase
        self._hash = hash((name, frequency, phase))

    @property
    def name(self) -> str:
        """Return the name of the frame."""
        return f"GenericFrame({self._name})"

    @property
    def frequency(self) -> float:
        """Return the initial frequency of the generic frame."""
        return self._frequency

    @property
    def phase(self) -> float:
        """Return the initial phase of the generic frame."""
        return 0 if self._phase is None else self._phase

    def __eq__(self, other: "GenericFrame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        name, frequency and phase.

        Args:
            other: The generic frame to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._name == other._name and self._frequency == other._frequency and self._phase == other._phase

    def __hash__(self):
        return self._hash

    @property
    def name(self) -> str:
        """Return the name of the generic frame."""
        return self._name


class QubitFrame(Frame):
    """A frame associated with the driving of a qubit."""
    def __init__(self, qubit_index: int):
        """Create ``QubitFrame``.

        Args:
            qubit_index: The index of the qubit represented by the frame.
        """
        self._validate_index(qubit_index)
        self._index = qubit_index
        super().__init__(self.name)

    @property
    def name(self) -> str:
        """Return the name of the qubit frame."""
        return f"QubitFrame{self.qubit_index}"

    @property
    def qubit_index(self):
        return self._index

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``identifier`` (index) is a negative integer.
        """
        pass
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")


class MeasurementFrame(Frame):
    """A frame associated with the measurement of a qubit."""
    def __init__(self, qubit_index: int):
        """Create ``MeasurementFrame``.

        Args:
            qubit_index: The index of the qubit represented by the frame.
        """
        self._validate_index(qubit_index)
        self._index = qubit_index
        super().__init__(self.name)

    @property
    def index(self) -> int:
        return int(self._index)

    @property
    def qubit_index(self) -> int:
        return int(self._index)

    @property
    def name(self) -> str:
        """Return the name of the frame."""
        return f"MeasurementFrame{self.qubit_index}"

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        pass
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")


class MixedFrame:
    """MixedFrame is a combination of a logical element and a frame."""
    def __init__(self, logical_element: LogicalElement, frame: Frame):
        """Create ``MixedFrame``.

        Args:
            logical_element: The logical element associated with the mixed frame.
            frame: The frame associated with the mixed frame.
        """
        self._logical_element = logical_element
        self._frame = frame
        self._hash = hash((self._logical_element, self._frame))

    @property
    def logical_element(self) -> LogicalElement:
        """Return the ``LogicalElement`` of this mixed frame."""
        return self._logical_element

    @property
    def frame(self) -> Frame:
        """Return the ``Frame`` of this mixed frame."""
        return self._frame

    @property
    def name(self) -> str:
        """Return the name of the mixed frame."""
        return f"MixedFrame({self.logical_element},{self.frame})"

    def __repr__(self):
        return self.name

    def __eq__(self, other: "MixedFrame") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same logical
        element and frame.

        Args:
            other: The mixed frame to compare to this one.

        Returns:
            True iff equal.
        """
        return self._logical_element == other._logical_element and self._frame == other._frame

    def __hash__(self):
        return self._hash


class CRMixedFrame(MixedFrame):
    """ A mixed frame for Cross Resonance control.

    ``CRMixedFrame`` is identical to ``MixedFrame`` but is devoted to the common case of cross
    resonance control. In this case the ``LogicalElement`` and ``Frame`` associated with the ``MixedFrame``
    are of types ``Qubit`` and ``QubitFrame`` respectively.

    ``CRMixedFrame`` and ``MixedFrame`` of the same elements will not only function in the same way,
    but will also be equal to one another. ``CRMixedFrame`` is used for improved readability
    and type hinting, as well as compilation validation.
    """
    def __init__(self, qubit: Qubit, qubit_frame: QubitFrame):
        """Create ``CRMixedFrame``.

        Args:
            qubit: The ``Qubit`` object associated with the mixed frame.
            qubit_frame: The ``QubitFrame`` associated with the mixed frame.
        """
        super().__init__(logical_element=qubit, frame=qubit_frame)

    @property
    def qubit(self) -> Qubit:
        """Return the ``Qubit`` object of this mixed frame."""
        return self.logical_element

    @property
    def qubit_frame(self) -> QubitFrame:
        """Return the ``QubitFrame`` of this mixed frame."""
        return self.frame
