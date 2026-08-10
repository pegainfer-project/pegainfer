#!/usr/bin/env python3
"""CPU mirror for the frozen FlashInfer/OpenInfer GDN state layout contract."""

from __future__ import annotations

from dataclasses import dataclass


UPSTREAM_ORDER = (0, 1, 2, 3)
OPENINFER_HKV_ORDER = (1, 0, 2, 3)


@dataclass(frozen=True)
class StateGeometry:
    heads: int
    key_dim: int
    value_dim: int
    sequences: int = 1

    @property
    def shape(self) -> tuple[int, int, int, int]:
        # CuTe axes addressed as gKV[k, v] after slicing head and sequence.
        return (self.key_dim, self.value_dim, self.heads, self.sequences)


def ordered_strides(
    shape: tuple[int, ...], order: tuple[int, ...]
) -> tuple[int, ...]:
    if sorted(order) != list(range(len(shape))):
        raise ValueError(f"order is not a permutation: {order}")
    strides = [0] * len(shape)
    stride = 1
    for axis in order:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def cute_state_offset(
    geometry: StateGeometry,
    *,
    head: int,
    key: int,
    value: int,
    sequence: int = 0,
    order: tuple[int, int, int, int] = OPENINFER_HKV_ORDER,
) -> int:
    coordinates = (key, value, head, sequence)
    for coordinate, extent in zip(coordinates, geometry.shape, strict=True):
        if coordinate < 0 or coordinate >= extent:
            raise IndexError(f"coordinate {coordinates} exceeds shape {geometry.shape}")
    strides = ordered_strides(geometry.shape, order)
    return sum(c * s for c, s in zip(coordinates, strides, strict=True))


def openinfer_hkv_offset(
    geometry: StateGeometry, *, head: int, key: int, value: int, sequence: int = 0
) -> int:
    return (
        ((sequence * geometry.heads + head) * geometry.key_dim + key)
        * geometry.value_dim
        + value
    )


def upstream_hvk_offset(
    geometry: StateGeometry, *, head: int, key: int, value: int, sequence: int = 0
) -> int:
    return (
        ((sequence * geometry.heads + head) * geometry.value_dim + value)
        * geometry.key_dim
        + key
    )


def asymmetric_value(head: int, key: int, value: int) -> int:
    return head * 100_000 + key * 100 + value


def first_wrong_mapping(
    geometry: StateGeometry,
) -> tuple[tuple[int, int, int], int, int] | None:
    memory = [0] * (
        geometry.sequences
        * geometry.heads
        * geometry.key_dim
        * geometry.value_dim
    )
    for head in range(geometry.heads):
        for key in range(geometry.key_dim):
            for value in range(geometry.value_dim):
                memory[
                    openinfer_hkv_offset(
                        geometry, head=head, key=key, value=value
                    )
                ] = asymmetric_value(head, key, value)

    for head in range(geometry.heads):
        for key in range(geometry.key_dim):
            for value in range(geometry.value_dim):
                expected = asymmetric_value(head, key, value)
                actual = memory[
                    cute_state_offset(
                        geometry,
                        head=head,
                        key=key,
                        value=value,
                        order=UPSTREAM_ORDER,
                    )
                ]
                if actual != expected:
                    return (head, key, value), expected, actual
    return None
