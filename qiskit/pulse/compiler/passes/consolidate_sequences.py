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

"""A sequence consolidation pass for Qiskit PulseIR compilation."""

from __future__ import annotations

from qiskit.pulse.compiler.basepasses import TransformationPass
from qiskit.pulse.ir import SequenceIR
from qiskit.pulse.transforms import AlignLeft, AlignRight, AlignSequential
from qiskit.pulse.exceptions import PulseCompilerError


class ConsolidateSequences(TransformationPass):
    """Consolidate unnecessary sequences in the IR.

    The pass traverses the ``SequenceIR``, and breaks down every nested ``SequenceIR`` which
    is unnecessary (i.e. a ``SequenceIR`` which has no effect on the final sequencing or scheduling
    of the instructions).

    The three types of unnecessary objects which are broken down:
    - ``SequenceIR`` with no elements (removed entirely).
    - ``SequenceIR`` of either left, right or sequential alignment with a single
        element (replaced by its element).
    - ``SequenceIR`` of sequential alignment nested in a ``SequenceIR`` of the
        same alignment (replaced by its elements).

    The pass will not update sequencing or scheduling information, and should be run before
    :class:`~qiskit.pulse.compiler.passes.SetSequence` and
    :class:`~qiskit.pulse.compiler.passes.SetSchedule`.
    """

    single_instruction_unnecessary_alignments = (AlignLeft, AlignRight, AlignSequential)

    def __init__(self):
        """Create new ConsolidateSequences pass"""
        super().__init__(target=None)

    def run(
        self,
        passmanager_ir: SequenceIR,
    ) -> SequenceIR:
        """Run the pass"""

        self._consolidate_recursion(passmanager_ir)
        return passmanager_ir

    def _consolidate_recursion(self, prog: SequenceIR) -> None:
        """Recursively consolidate sequences

        Arguments:
            prog: The IR object to undergo consolidation.
        """
        if any(
            prog.time_table[x] is not None for x in prog.sequence.node_indices() if x not in (0, 1)
        ):
            raise PulseCompilerError(
                "Can not consolidate sequences in an object which is already scheduled."
                "Run ConsolidateSequences before SetSchedule."
            )

        for ind in prog.sequence.node_indices():
            if ind in (0, 1):
                continue
            if isinstance(sub_prog := prog.sequence.get_node_data(ind), SequenceIR):
                if len(sub_prog.elements()) == 0:
                    prog.sequence.remove_node_retain_edges(ind)
                    continue

                self._consolidate_recursion(sub_prog)
                if (
                    isinstance(sub_prog.alignment, self.single_instruction_unnecessary_alignments)
                    and len(sub_prog.elements()) == 1
                ):
                    prog.sequence[ind] = sub_prog.elements()[0]
                elif isinstance(sub_prog.alignment, AlignSequential) and isinstance(
                    prog.alignment, AlignSequential
                ):

                    def edge_map(_x, _y, _node):
                        if _y == _node:
                            return 0
                        if _x == _node:
                            return 1
                        return None

                    nodes_mapping = prog.sequence.substitute_node_with_subgraph(
                        ind, sub_prog.sequence, lambda x, y, _: edge_map(x, y, ind)
                    )
                    if (initial_time := prog.time_table[ind]) is not None:
                        for old_node in nodes_mapping.keys():
                            if old_node not in (0, 1):
                                prog.time_table[nodes_mapping[old_node]] = (
                                    initial_time + sub_prog.time_table[old_node]
                                )
                        del prog.time_table[ind]

                    prog.sequence.remove_node_retain_edges(nodes_mapping[0])
                    prog.sequence.remove_node_retain_edges(nodes_mapping[1])

    def __hash__(self):
        return hash((self.__class__.__name__,))

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__
