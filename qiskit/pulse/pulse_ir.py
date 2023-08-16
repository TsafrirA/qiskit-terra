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

# pylint: disable=cyclic-import

"""
=========
Pulse IR
=========

"""

from typing import Union, Optional, List
from abc import ABCMeta, abstractmethod

import numpy as np

from qiskit.pulse.exceptions import PulseError

from qiskit.pulse import (
    LogicalElement,
    Frame,
    MixedFrame,
    SymbolicPulse,
    Waveform,
    Qubit,
    MemorySlot
)
from qiskit.pulse.transforms import AlignmentKind


class BaseIRInstruction(metaclass=ABCMeta):
    """Base class for PulseIR instruction."""

    @abstractmethod
    def __init__(
        self,
        duration: int,
        operand,
        t0: Optional[int] = None
    ):
        """Base class for PulseIR instruction.

        Args:
            duration: Duration of the instruction (in terms of system ``dt``).
            operand: The operation of the instruction.
            t0 (Optional): Starting time of the instruction. Defaults to ``None``

        Raises:
            PulseError: if ``duration`` is not non-negative integer.
            PulseError: if ``t0`` is not ``None`` and not non-negative integer.

        """

        if not isinstance(duration, (int, np.integer)) or duration < 0:
            raise PulseError("duration must be a non-negative integer.")
        if t0 is not None and not isinstance(t0, (int, np.integer)) or t0 < 0:
            raise PulseError("t0 must be a non-negative integer.")

        self._duration = duration
        self._operand = operand
        self._t0 = t0

    @property
    def operand(self):
        return self._operand

    @property
    def duration(self) -> int:
        return self._duration

    @property
    def t0(self) -> int:
        return self._t0

    @t0.setter
    def t0(self, value: int):
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise PulseError("t0 must be a non-negative integer")
        self._t0 = value

    def shift_t0(self, value: int):
        if self._t0 is None:
            raise PulseError("Can not shift t0 of an untimed instruction")
        else:
            self.t0 = self.t0 + value

    @property
    def final_time(self) -> int:
        return self._t0 + self._duration


class GenericInstruction(BaseIRInstruction):
    """PulseIR Generic Instruction"""
    allowed_types = ["Play", "Delay", "SetFrequency", "ShiftFrequency", "SetPhase", "ShiftPhase"]

    def __init__(
        self,
        instruction_type: str,
        operand,
        logical_element: Optional[LogicalElement] = None,
        frame: Optional[Frame] = None,
        t0: Optional[int] = None
    ):
        """PulseIR Generic Instruction

        Args:
            instruction_type: The type of instruction.
            operand: The operand describing the operation of the instruction.
            logical_element (Optional): The logical element associated with the instruction. Defaults to ``None``
            frame (Optional): The frame associated with the instruction. Defaults to ``None``
            t0 (Optional): Starting time of the instruction. Defaults to ``None``

        Raises:
            PulseError: if ``duration`` is not non-negative integer.
            PulseError: if ``t0`` is not ``None`` and not non-negative integer.

        """
        duration = self._validate_instruction(instruction_type, operand, logical_element, frame)
        self._instruction_type = instruction_type
        self._logical_element = logical_element
        self._frame = frame
        super().__init__(duration, operand, t0)

    def _validate_instruction(self, instruction_type, operand, logical_element, frame):
        if instruction_type not in self.__class__.allowed_types:
            raise PulseError(f"{instruction_type} is not a recognized instruction")

        if instruction_type == "Delay":
            if not isinstance(operand, (int, np.integer)) or operand < 0:
                raise PulseError("The operand of a Delay instruction must be a non-negative integer.")
            if logical_element is None:
                raise PulseError("Delay instruction must have an associated logical element.")
            duration = operand

        elif instruction_type == "Play":
            if not isinstance(operand, (SymbolicPulse, Waveform)):
                raise PulseError(f"Play instruction is incompatible with operand of type {type(operand)}.")
            if logical_element is None or frame is None:
                raise PulseError("Play instruction must have an associated logical element and frame.")
            duration = operand.duration

        elif instruction_type in ["SetFrequency", "ShiftFrequency", "SetPhase", "ShiftPhase"]:
            if isinstance(operand, (int, np.integer)):
                operand = float(operand)
            if not isinstance(operand, float):
                raise PulseError("The operand of a Set/Shift Frequency/Phase instruction must be a float.")
            if frame is None:
                raise PulseError("Set/Shift Frequency/Phase instruction instruction must have an associated frame.")
            duration = 0

        return duration

    @property
    def instruction_type(self) -> str:
        return self._instruction_type

    @property
    def logical_element(self) -> LogicalElement:
        return self._logical_element

    @property
    def frame(self) -> Frame:
        return self._frame

    def __repr__(self):
        return f"{self._instruction_type}" \
               "(" \
               f"operand={self._operand}," \
               f"LogicalElement={self._logical_element}," \
               f"frame={self._logical_element}," \
               f"duration={self._duration}," \
               f"t0={self._t0}" \
               ")"


class AcquireInstruction(BaseIRInstruction):
    def __init__(
        self,
        qubit: Qubit,
        memory_slot: MemorySlot,
        duration: int,
        t0: Optional[int] = None
    ):
        self._qubit = qubit
        self._memory_slot = memory_slot
        operand = duration
        super().__init__(duration, operand, t0)

    @property
    def qubit(self) -> Qubit:
        return self._qubit

    @property
    def memory_slot(self) -> MemorySlot:
        return self._memory_slot

    def __repr__(self):
        return "Acquire" \
               "(" \
               f"qubit={self._qubit}," \
               f"memory_slot={self._memory_slot}," \
               f"duration={self._duration}," \
               f"t0={self._t0}" \
               ")"


class PulseIR:
    """
    """

    def __init__(self, alignment: Optional[AlignmentKind] = None):
        """Sets default values to PulseIR"""
        self._elements = []
        self._alignment = alignment

    @property
    def t0(self):
        elements_t0s = [element.t0 for element in self._elements]
        if any([x is None for x in elements_t0s]):
            return None
        else:
            return min(elements_t0s)

    @property
    def final_time(self):
        elements_final_times = [element.final_time for element in self._elements]
        if any([x is None for x in elements_final_times]):
            return None
        else:
            return max(elements_final_times)

    @property
    def elements(self) -> List[Union[GenericInstruction, AcquireInstruction, "PulseIR"]]:
        return self._elements

    def alignment(self) -> AlignmentKind:
        return self._alignment

    def add_element(
        self,
        element: Union[GenericInstruction, AcquireInstruction, "PulseIR", List[
            Union[GenericInstruction, AcquireInstruction, "PulseIR"]]],
    ):
        """Adds IR element (instruction or PulseIR) or list thereof to the object.

        Args:
            element: The instruction or nested PulseIR object to add.
        """
        if type(element) is not list:
            element = [element]

        self._elements.extend(element)

    def shift_t0(self, value: int):
        for element in self._elements:
            element.shift_t0(value)

    def frames(self) -> List[Frame]:
        frames = set()
        for element in self._elements:
            if isinstance(element, GenericInstruction) and element.frame is not None:
                frames.add(element.frame)
            elif isinstance(element, PulseIR):
                frames |= element.frames()
        return frames

    def logical_elements(self) -> List[LogicalElement]:
        """Recursively list all logical elements in the IR"""
        logical_elements = set()
        for element in self._elements:
            if isinstance(element, GenericInstruction) and element.logical_element is not None:
                logical_elements.add(element.logical_element)
            elif isinstance(element, AcquireInstruction):
                logical_elements.add(element.qubit)
            else:
                logical_elements |= element.logical_elements()
        return logical_elements

    def mixed_frames(self) -> List[MixedFrame]:
        """Recursively list all mixed frames in the IR"""
        mixed_frames = set()
        for element in self._elements:
            if isinstance(element, GenericInstruction):
                if element.logical_element is None or element.frame is None:
                    raise PulseError("PulseIR contains non-broadcasted instructions, and its"
                                     "mixed frames are not properly defined")
                mixed_frames.add(MixedFrame(element.logical_element, element.frame))
            elif isinstance(element, PulseIR):
                mixed_frames |= element.mixed_frames()
        return mixed_frames

    def flatten(self) -> "PulseIR":
        """Recursively flatten the IR into a single block"""
        flat_ir = PulseIR()
        for element in self._elements:
            if isinstance(element, BaseIRInstruction):
                if element.t0 is None:
                    raise PulseError("Can not flatten an untimed PulseIR")
                flat_ir.add_element(element)
            else:
                flat_ir.add_element(element.flatten().elements)
        return flat_ir

    def get_instructions_by_mixed_frame(self, mixed_frame: MixedFrame) -> List[GenericInstruction]:
        """Recursively get instructions associated with a given mixed frame"""
        instructions = []
        for element in self._elements:
            if isinstance(element, GenericInstruction):
                if mixed_frame == MixedFrame(element.logical_element, element.frame):
                    instructions.append(element)
            elif isinstance(element, PulseIR):
                instructions.extend(element.get_instructions_by_mixed_frame(mixed_frame))
        return instructions

    def get_acquire_instructions(self, qubit: Optional[Qubit] = None) -> List[GenericInstruction]:
        """Recursively get acquire instructions, optionally for a given qubit"""
        instructions = []
        for element in self._elements:
            if isinstance(element, AcquireInstruction) and (qubit is None or element.qubit == qubit):
                instructions.append(element)
            elif isinstance(element, PulseIR):
                instructions.extend(element.get_acquire_instructions(qubit))
        return instructions

    def sort_by_t0(self):
        """Sort the elements of the object by ``t0`` in place"""
        t0s = [element.t0 for element in self._elements]
        self._elements = [self._elements[i] for i in np.argsort(t0s)]

    def __repr__(self):
        repr = f"[alignment={self._alignment},["
        for element in self._elements:
            repr += str(element) + ", "
        repr = repr[:-2] + "]]"
        return repr
