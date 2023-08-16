from qiskit.pulse import (
    Qubit,
    QubitFrame,
    MeasurementFrame,
    Drag,
    MemorySlot,
    GaussianSquare,
    GenericFrame,
)

from qiskit.pulse.pulse_ir import (
    PulseIR,
    AcquireInstruction,
    GenericInstruction,
)


def measure_qubit(qubit_ind, memslot_ind) -> PulseIR:
    pulse_ir = PulseIR()
    pulse = GaussianSquare(duration=3520, sigma=64, width=3264, amp=0.35, angle=-2.1578425359693685)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse,
        logical_element=Qubit(qubit_ind),
        frame=MeasurementFrame(qubit_ind),
        t0=0,
    )
    pulse_ir.add_element(inst)
    inst = AcquireInstruction(
        qubit=Qubit(qubit_ind),
        memory_slot=MemorySlot(memslot_ind),
        duration=3520,
        t0=0,
    )
    pulse_ir.add_element(inst)
    return pulse_ir


def x_gate_and_measure() -> PulseIR:
    """X gate on qubit 1 and measurement to memory slot 3. No delays"""
    pulse_ir = PulseIR()
    pulse = Drag(duration=256, sigma=64, beta=3.2599789746223267, amp=0.19901277069390055, angle=0.0)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse,
        logical_element=Qubit(1),
        frame=QubitFrame(1),
        t0=0,
    )
    pulse_ir.add_element(inst)

    meas_pulseir = measure_qubit(1, 3)
    meas_pulseir.shift_t0(256)
    pulse_ir.add_element(meas_pulseir)

    return pulse_ir


def ecr_and_measure() -> PulseIR:
    """echoed cross resonance on qubits 3 and 4. No delays"""
    pulse_ir = PulseIR()

    block = PulseIR()
    pulse1 = GaussianSquare(duration=1072, sigma=64, width=816, amp=0.07559986452478391, angle=0.02257062876253748)
    inst1 = GenericInstruction(
        instruction_type="Play",
        operand=pulse1,
        logical_element=Qubit(3),
        frame=QubitFrame(3),
        t0=0,
    )

    pulse2 = GaussianSquare(duration=1072, sigma=64, width=816, amp=0.11540735255558918, angle=-1.1567559579400704)
    inst2 = GenericInstruction(
        instruction_type="Play",
        operand=pulse2,
        logical_element=Qubit(4),
        frame=QubitFrame(3),
        t0=0,
    )
    block.add_element([inst1, inst2])
    pulse_ir.add_element(block)

    pulse = Drag(duration=256, sigma=64, beta=1.6849082066036696, amp=0.1866528287038667, angle=0.0)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse,
        logical_element=Qubit(4),
        frame=QubitFrame(4),
        t0=1072,
    )
    pulse_ir.add_element(inst)

    block = PulseIR()
    pulse1 = GaussianSquare(duration=1072, sigma=64, width=816, amp=0.07559986452478391, angle=-3.119022024827256)
    inst1 = GenericInstruction(
        instruction_type="Play",
        operand=pulse1,
        logical_element=Qubit(3),
        frame=QubitFrame(3),
        t0=1328,
    )

    pulse2 = GaussianSquare(duration=1072, sigma=64, width=816, amp=0.11540735255558918, angle=1.9848366956497228)
    inst2 = GenericInstruction(
        instruction_type="Play",
        operand=pulse2,
        logical_element=Qubit(4),
        frame=QubitFrame(3),
        t0=1328,
    )
    block.add_element([inst1, inst2])
    pulse_ir.add_element(block)

    final_time = pulse_ir.final_time

    meas_block = PulseIR()
    meas_block.add_element(measure_qubit(3, 3))
    meas_block.add_element(measure_qubit(4, 4))
    meas_block.shift_t0(final_time)
    pulse_ir.add_element(meas_block)

    return pulse_ir


def general_example() -> PulseIR:
    """Just a bunch of random instructions"""

    pulse_ir = PulseIR()
    pulse = Drag(duration=1072, sigma=64, beta=0.01, amp=0.07559986452478391, angle=0.02257062876253748)

    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse,
        logical_element=Qubit(2),
        frame=QubitFrame(2),
        t0=0,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse,
        logical_element=Qubit(1),
        frame=QubitFrame(2),
        t0=160,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="ShiftFrequency",
        operand=102.6,
        logical_element=Qubit(2),
        frame=MeasurementFrame(2),
        t0=64,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="SetFrequency",
        operand=10**9,
        logical_element=Qubit(1),
        frame=QubitFrame(1),
        t0=32,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="Delay",
        operand=176,
        logical_element=Qubit(1),
        frame=QubitFrame(2),
        t0=98,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse,
        logical_element=Qubit(1),
        frame=MeasurementFrame(1),
        t0=4000,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="ShiftPhase",
        operand=0.5,
        logical_element=Qubit(1),
        frame=QubitFrame(1),
        t0=8000,
    )
    pulse_ir.add_element(inst)
    return pulse_ir


def qudit_example() -> PulseIR:
    """Using generic frame to control different transitions of a single qubit"""
    pulse_ir = PulseIR()
    frame = GenericFrame("1-2 transition", 100.2)
    frame1 = GenericFrame("2-3 transition", 50.1)
    pulse1 = Drag(duration=1072, sigma=64, beta=0.01, amp=0.07559986452478391, angle=0.02257062876253748)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse1,
        logical_element=Qubit(2),
        frame=QubitFrame(2),
        t0=0,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse1,
        logical_element=Qubit(2),
        frame=frame,
        t0=1072,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="Play",
        operand=pulse1,
        logical_element=Qubit(2),
        frame=frame1,
        t0=1072*2,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="SetFrequency",
        operand=frame.frequency,
        logical_element=Qubit(2),
        frame=frame,
        t0=0,
    )
    pulse_ir.add_element(inst)
    inst = GenericInstruction(
        instruction_type="SetFrequency",
        operand=frame1.frequency,
        logical_element=Qubit(2),
        frame=frame1,
        t0=0,
    )
    pulse_ir.add_element(inst)

    return pulse_ir

print(general_example())
