#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
import warnings
from importlib.resources import files

import numpy as np
import numpy.typing as npt

from . import datasets

datasets_directory = files(datasets)


class StabilizerStates:
    """A container class for stabilizer states"""

    def __init__(self, num_qubits: int, format: str = "any"):
        """
        Initialize a StabilizerStates instance.

        :param num_qubits: The stabilizer state dataset to load.
        :param format: The representation format for the states. Available options are "any", "real", "ternary".

        :raises ValueError: if the stabilizer state dataset is missing.
        """

        if format == "any":
            search_order = ["", "_ternary"]
        elif format == "real" or format == "ternary":
            # Ternary datasets are quicker to load, so try those first
            search_order = ["_ternary", ""]

        for dataset in search_order:
            try:
                file = datasets_directory.joinpath(f"S{num_qubits}{dataset}.npy")
                self._states = np.load(file)
                break
            except FileNotFoundError:
                self._states = None
                continue

        if self._states is None:
            raise ValueError(f"Missing stabilizer state dataset for {num_qubits} qubits.")

        if format == "any":
            full_count = StabilizerStates.count(num_qubits)
            dataset_count = len(self._states)
            if dataset_count != full_count:
                warnings.warn(
                    f"The dataset loaded ({file.name}) is a partial dataset ({dataset_count} vs. {full_count})"
                )

        if format == "real" or format == "ternary":
            if "ternary" not in str(file):
                # Remove imaginary states from the dataset if we loaded the full dataset
                self._states = np.real(self._states[np.all(np.isreal(self._states), axis=1)])

            if format == "real" and "ternary" in str(file):
                # If we loaded a ternary dataset, then we need to normalize to get back to reals
                warnings.warn("Real dataset requested, but ternary was loaded, so we must normalize; might be slow.")
                self._states = np.apply_along_axis(StabilizerStates._normalize, 1, self._states.astype(float))
            elif format == "ternary":
                # We are relying on the previous filter to remove all imaginary states first
                self._states = np.real(np.sign(self._states)).astype(np.int8)

    @staticmethod
    def count(num_qubits: int) -> int:
        """
        Counts the number of stabilizer states according to Proposition 2 from Aaronson / Gottesman in "Improved
        Simulation of Stabilizer Circuits" (https://arxiv.org/abs/quant-ph/0406196)

        :param num_qubits: The number of qubits to use for counting the number of pure stabilizer states.
        :return: The number of pure stabilizer states for ``num_qubits``.
        """
        count = 2**num_qubits
        for k in range(0, num_qubits):
            count *= 2 ** (num_qubits - k) + 1
        return count

    @staticmethod
    def _normalize(v: npt.NDArray) -> npt.NDArray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def validate(self, show=True) -> bool:
        """
        Validate the stabilizer states.

        :param show: Print the results of each test.
        :return: True if stabilizer states are valid; Otherwise, False.

        García et al. in https://arxiv.org/abs/1711.07848 establish the following amplitude properties of stabilizer
        states (Corollary 2):
            1. number of non-zero amplitudes (support) is a power of two
            2. they are unbiased, and every non-zero amplitude is ±1/√|s| or ±i/√|s|, where |s| is the support
            3. the number of imaginary amplitudes is either zero or half the number of non-zero amplitudes
            4. the number of negative amplitudes is either zero or a power of two
            (we do not check property 5)
        """
        num_qubits = int(np.log2(self.shape[1]))
        full_count = StabilizerStates.count(num_qubits)
        states = self._states
        dataset_count = len(states)
        is_full_set = full_count == dataset_count

        if show:
            print(f"Validation for {num_qubits}-qubit stabilizer dataset:")
            print(f"  Full count of stabilizer states ({full_count}): {'✅' if is_full_set else '❌'}")

        all_unique = len(np.unique(states, axis=1)) == dataset_count

        if show:
            print(f"  All states unique: {'✅' if all_unique else '❌'}")

        # Check property 1
        support = np.count_nonzero(states, axis=1)
        nonzero_power_of_two = np.all(np.mod(np.log2(support), 1) == 0)

        if show:
            print(
                f"  Non-zero amplitude count for each state is a power of two: {'✅' if nonzero_power_of_two else '❌'}"
            )

        # Check property 2
        amplitudes = 1 / np.sqrt(support[:, None])
        nonzero_amps_match = np.isclose(states, amplitudes)
        nonzero_amps_match = np.where(nonzero_amps_match, nonzero_amps_match, np.isclose(states, -amplitudes))
        nonzero_amps_match = np.where(nonzero_amps_match, nonzero_amps_match, np.isclose(states, 1j * amplitudes))
        nonzero_amps_match = np.where(nonzero_amps_match, nonzero_amps_match, np.isclose(states, -1j * amplitudes))
        nonzero_amps_match = np.where(nonzero_amps_match, nonzero_amps_match, np.isclose(states, 0))
        nonzero_amps_match = np.all(nonzero_amps_match)

        if show:
            print(
                f"  Non-zero amplitude count for each state is either ±1/√|s| or ±i/√|s| (|s| is support): "
                f"{'✅' if nonzero_amps_match else '❌'}"
            )

        # Check property 3
        num_imaginary = np.count_nonzero(np.iscomplex(states), axis=1)
        imaginary_amps_half_support = num_imaginary == support // 2
        imaginary_amps_half_support = np.where(
            imaginary_amps_half_support, imaginary_amps_half_support, num_imaginary == 0
        )
        imaginary_amps_half_support = np.all(imaginary_amps_half_support)

        if show:
            print(
                f"  Imaginary amplitude count for each state is zero or half the support: "
                f"{'✅' if imaginary_amps_half_support else '❌'}"
            )

        # Check property 4
        num_negative_signs = np.count_nonzero(np.sign(states) == -1, axis=1)
        negative_amps_power_of_two = np.all(
            np.mod(np.round(np.log2(num_negative_signs, where=num_negative_signs > 0)), 1) == 0
        )

        if show:
            print(
                f"  Negative amplitude count for each state is zero or a power of two: "
                f"{'✅' if negative_amps_power_of_two else '❌'}"
            )

        return (
            is_full_set
            and all_unique
            and nonzero_power_of_two
            and nonzero_amps_match
            and imaginary_amps_half_support
            and negative_amps_power_of_two
        )

    @property
    def shape(self) -> tuple:
        """Return the shape of the dataset"""
        return self._states.shape

    def __iter__(self):
        def iterator():
            for state in self._states:
                yield state

        return iterator()

    def __len__(self):
        return len(self._states)

    def __getitem__(self, item):
        return self._states[item]

    def __str__(self):
        return str(self._states)
