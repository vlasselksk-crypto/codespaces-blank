"""FlatBuffers helpers for SlothAC TickDataSequence."""

from __future__ import annotations

import flatbuffers
from app.flatbuffers.slothac import TickData, TickDataSequence


def parse_tickdata_sequence(body: bytes) -> list[list[float]]:
    """Parse a TickDataSequence FlatBuffers blob.

    Returns a list of tick rows, each with 8 float values.
    """

    seq = TickDataSequence.TickDataSequence.GetRootAsTickDataSequence(body, 0)
    length = seq.TicksLength()
    ticks: list[list[float]] = []
    for i in range(length):
        offset = seq.Ticks(i)
        if not offset:
            continue
        td = TickData.TickData()
        td.Init(body, offset)
        ticks.append([
            td.F0(),
            td.F1(),
            td.F2(),
            td.F3(),
            td.F4(),
            td.F5(),
            td.F6(),
            td.F7(),
        ])

    return ticks


def build_tickdata_sequence(rows: list[list[float]]) -> bytes:
    """Build a TickDataSequence FlatBuffers payload for tests or clients."""

    builder = flatbuffers.Builder(0)

    # build each TickData table and store offsets
    offsets = []
    for r in rows:
        if len(r) != 8:
            raise ValueError("Each tick row must have exactly 8 floats")
        TickData.TickDataStart(builder)
        TickData.TickDataAddF0(builder, float(r[0]))
        TickData.TickDataAddF1(builder, float(r[1]))
        TickData.TickDataAddF2(builder, float(r[2]))
        TickData.TickDataAddF3(builder, float(r[3]))
        TickData.TickDataAddF4(builder, float(r[4]))
        TickData.TickDataAddF5(builder, float(r[5]))
        TickData.TickDataAddF6(builder, float(r[6]))
        TickData.TickDataAddF7(builder, float(r[7]))
        offsets.append(TickData.TickDataEnd(builder))

    # Create the vector of TickData offsets
    TickDataSequence.TickDataSequenceStartTicksVector(builder, len(offsets))
    # vector elements must be added in reverse order
    for off in reversed(offsets):
        builder.PrependUOffsetTRelative(off)
    ticks_vec = builder.EndVector()

    # Finish sequence
    TickDataSequence.TickDataSequenceStart(builder)
    TickDataSequence.TickDataSequenceAddTicks(builder, ticks_vec)
    root = TickDataSequence.TickDataSequenceEnd(builder)
    builder.Finish(root)
    return bytes(builder.Output())
