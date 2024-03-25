# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test SetSequence"""
from test import QiskitTestCase
from ddt import ddt, named_data, unpack

from qiskit.pulse import (
    Constant,
    Play,
)

from qiskit.pulse.ir import (
    SequenceIR,
)

from qiskit.pulse.model import QubitFrame, Qubit, MixedFrame
from qiskit.pulse.transforms import (
    AlignLeft,
    AlignRight,
    AlignSequential,
    AlignFunc,
    AlignEquispaced,
)
from qiskit.pulse.compiler import MapMixedFrame, ConsolidateSequences, SetSequence
from .utils import PulseIrTranspiler


@ddt
class TestConsolidateSequences(QiskitTestCase):
    """ConsolidateSequences tests"""

    def test_equating(self):
        """Test pass equating"""
        self.assertTrue(ConsolidateSequences() == ConsolidateSequences())
        self.assertFalse(ConsolidateSequences() == MapMixedFrame())

    def setUp(self):
        super().setUp()
        self._pm = PulseIrTranspiler([MapMixedFrame(), SetSequence(), ConsolidateSequences()])

    @named_data(
        ["align_left", AlignLeft()],
        ["align_left", AlignRight()],
        ["align_left", AlignSequential()],
        ["align_equispaced", AlignEquispaced(1000)],
        ["align_func", AlignFunc(1000, lambda x: x)],
    )
    @unpack
    def test_empty_sub_sequence(self, alignment):
        """test that empty sub-sequence is removed, regardless of alignment"""

        ir_example = SequenceIR(AlignSequential())
        ir_example.append(SequenceIR(alignment))
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(len(ir_example.elements()), 1)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 2)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    @named_data(
        ["align_left", AlignLeft()], ["align_left", AlignRight()], ["align_left", AlignSequential()]
    )
    @unpack
    def test_single_instruction_sub_sequence_removed_alignment(self, alignment):
        """test that single instruction sub-sequence is broken down, if in allowed alignments"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        inst = Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(2), QubitFrame(2)))
        sub_block = SequenceIR(alignment)
        sub_block.append(inst)
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        self.assertEqual(len(ir_example.elements()), 2)
        self.assertEqual(ir_example.elements()[1], inst)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 4)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((0, 3) in edge_list)
        self.assertTrue((2, 1) in edge_list)
        self.assertTrue((3, 1) in edge_list)

    @named_data(
        ["align_equispaced", AlignEquispaced(1000)],
        ["align_func", AlignFunc(1000, lambda x: x)],
    )
    @unpack
    def test_single_instruction_sub_sequence_non_removed_alignment(self, alignment):
        """test that single instruction sub-sequence is not broken down, if not in allowed alignments"""

        ir_example = SequenceIR(AlignRight())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        sub_block = SequenceIR(alignment)
        sub_block.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(2), QubitFrame(2))))
        ir_example.append(sub_block)

        ref = PulseIrTranspiler([MapMixedFrame(), SetSequence()]).run(ir_example.copy())
        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example, ref)

    def test_nested_sequential_alignments(self):
        """test that nested sequential alignments are broken down"""
        sub_block = SequenceIR(AlignSequential())
        inst1 = Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(2), QubitFrame(2)))
        inst2 = Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(3), QubitFrame(3)))
        sub_block.append(inst1)
        sub_block.append(inst2)

        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(sub_block)
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))

        ir_example = self._pm.run(ir_example)
        self.assertEqual(ir_example.sequence[7], inst1)
        self.assertEqual(ir_example.sequence[8], inst2)
        edge_list = ir_example.sequence.edge_list()
        self.assertEqual(len(edge_list), 5)
        self.assertTrue((0, 2) in edge_list)
        self.assertTrue((2, 7) in edge_list)
        self.assertTrue((7, 8) in edge_list)
        self.assertTrue((8, 4) in edge_list)
        self.assertTrue((4, 1) in edge_list)

    def test_recursion(self):
        """test that the pass is applied recursively"""
        sub_block = SequenceIR(AlignSequential())
        sub_block.append(SequenceIR(AlignLeft()))

        ir_example = SequenceIR(AlignSequential())
        ir_example.append(Play(Constant(100, 0.1), mixed_frame=MixedFrame(Qubit(0), QubitFrame(1))))
        ir_example.append(sub_block)

        ir_example = self._pm.run(ir_example)
        self.assertEqual(len(ir_example.elements()), 1)
