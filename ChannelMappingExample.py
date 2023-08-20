from ExamplesForJosh import x_gate_and_measure,ecr_and_measure
from qiskit.pulse import MixedFrame, Qubit, QubitFrame, MeasurementFrame
from qiskit.pulse.pulse_ir import PulseIR
from typing import Dict
from qiskit.providers.fake_provider import FakeJakarta


def map_mixed_frames(ir: PulseIR, backend_configuration) -> Dict:
    mixed_frames = ir.mixed_frames()
    mf_channel_mapping = dict()
    # First Step: Native mapping
    for mixed_frame in mixed_frames:
        mf_channel_mapping[mixed_frame] = None
        # Native mapping only works with qubits and either QubitFrame or MeasurementFrame
        if isinstance(mixed_frame.logical_element, Qubit):
            ind1 = mixed_frame.logical_element.index
            if isinstance(mixed_frame.frame, QubitFrame):
                ind2 = mixed_frame.frame.qubit_index
                if ind1 == ind2:
                    mf_channel_mapping[mixed_frame] = "d" + str(ind1)
                else:
                    try:
                        chan_name = backend_configuration.control_channels[(ind1, ind2)][0].name
                        mf_channel_mapping[mixed_frame] = chan_name
                    except KeyError:
                        pass
            elif isinstance(mixed_frame.frame, MeasurementFrame):
                ind2 = mixed_frame.frame.qubit_index
                if ind1 == ind2:
                    mf_channel_mapping[mixed_frame] = "m" + str(ind1)

    # Second Step: Use unused channels
    if any([x is None for x in mf_channel_mapping.values()]):
        raise NotImplementedError

    return mf_channel_mapping


backend_config = FakeJakarta().configuration()

ir = x_gate_and_measure()
mapping = map_mixed_frames(ir, backend_config)
print(mapping)

ir = ecr_and_measure()
mapping = map_mixed_frames(ir, backend_config)
print(mapping)
