"""CMA-ES state inheritance for Lamarckian evolution.

This module provides infrastructure for inheriting CMA-ES covariance matrices
and step sizes from parents to offspring when morphologies change.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.random import Generator

from ariel.ec.a001 import Individual
from ariel.ec.strategies.weight_inheritance import tree_distance


def _calculate_total_weights(layer_sizes: list[int]) -> int:
    """Calculate total number of weights including biases.

    For each layer transition i -> i+1:
    - Weights: layer_sizes[i] * layer_sizes[i+1]
    - Biases: layer_sizes[i+1]

    Args:
        layer_sizes: Neural network architecture [input, hidden..., output]

    Returns:
        Total number of weights and biases
    """
    total = 0
    for i in range(len(layer_sizes) - 1):
        # Weights for this layer
        total += layer_sizes[i] * layer_sizes[i + 1]
        # Biases for this layer
        total += layer_sizes[i + 1]
    return total


@dataclass
class CMAESState:
    """Complete CMA-ES state for inheritance.

    Stores both the full nevergrad optimizer state (for complete restoration)
    and extracted key components (for convenient access and adaptation).

    Attributes:
        nevergrad_state: Full pickled state dict from nevergrad optimizer.dump()
        covariance_matrix: The learned covariance matrix (C)
        sigma: The step size
        mean: The current search center
        layer_sizes: Neural network architecture [input, hidden..., output]
        condition_number: Condition number of covariance matrix
    """

    nevergrad_state: dict[str, Any] | None
    covariance_matrix: np.ndarray
    sigma: float
    mean: np.ndarray
    layer_sizes: list[int]
    condition_number: float | None = None

    @property
    def dimension(self) -> int:
        """Total number of weights (dimension of search space)."""
        return len(self.mean)

    def compute_condition_number(self) -> float:
        """Compute condition number of covariance matrix."""
        eigenvalues = np.linalg.eigvalsh(self.covariance_matrix)
        if np.min(eigenvalues) <= 0:
            return np.inf
        return np.max(eigenvalues) / np.min(eigenvalues)


def extract_cmaes_state_from_nevergrad(
    optimizer: Any,
    layer_sizes: list[int],
) -> CMAESState:
    """Extract CMA-ES state from a nevergrad CMA optimizer.

    Args:
        optimizer: Nevergrad CMA optimizer after optimization
        layer_sizes: Neural network architecture

    Returns:
        CMAESState with full state extracted
    """
    # Get the underlying CMA-ES optimizer
    cma_es = optimizer.optim.es

    # Extract key components
    covariance = np.array(cma_es.C)  # Covariance matrix
    sigma = float(cma_es.sigma)  # Step size
    mean = np.array(cma_es.mean)  # Current center

    # Note: We don't use nevergrad's dump() here because it requires a filepath
    # and we're handling serialization ourselves. Set to None for now.
    nevergrad_state = None

    # Create state object
    state = CMAESState(
        nevergrad_state=nevergrad_state,
        covariance_matrix=covariance,
        sigma=sigma,
        mean=mean,
        layer_sizes=layer_sizes,
    )

    # Compute condition number
    state.condition_number = state.compute_condition_number()

    return state


def create_default_cmaes_state(
    layer_sizes: list[int],
    sigma_init: float = 1.0,
) -> CMAESState:
    """Create a default CMA-ES state with identity covariance.

    Used for individuals that skip CMA-ES optimization (< 4 actuators)
    or when no parent state is available.

    Args:
        layer_sizes: Neural network architecture
        sigma_init: Initial step size

    Returns:
        Default CMAESState with identity covariance
    """
    # Calculate total number of weights (including biases)
    total_weights = _calculate_total_weights(layer_sizes)

    # Create identity covariance and zero mean
    covariance = np.eye(total_weights)
    mean = np.zeros(total_weights)

    return CMAESState(
        nevergrad_state=None,  # No nevergrad state for default
        covariance_matrix=covariance,
        sigma=sigma_init,
        mean=mean,
        layer_sizes=layer_sizes,
        condition_number=1.0,  # Identity matrix has condition number 1
    )


def adapt_cmaes_state_to_morphology(
    parent_state: CMAESState,
    parent_layer_sizes: list[int],
    offspring_layer_sizes: list[int],
    sigma_init: float,
    covariance_mode: Literal["adaptive", "reset", "preserve"] = "adaptive",
    sigma_mode: Literal["blend", "reset", "keep", "adaptive"] = "blend",
    rng: Generator | None = None,
) -> CMAESState:
    """Adapt parent's CMA-ES state to offspring's morphology.

    Handles changes in neural network architecture when morphology changes.

    Covariance adaptation modes:
        - "adaptive": Preserve correlations for existing weights, identity for new
        - "reset": Full identity matrix (no inheritance)
        - "preserve": Scale parent covariance to new size (experimental)

    Sigma adaptation modes:
        - "blend": Blend parent sigma and sigma_init based on proportion of new weights
        - "reset": Always use sigma_init (no inheritance)
        - "keep": Always use parent sigma (full inheritance)
        - "adaptive": Custom blending logic (currently same as "blend")

    Args:
        parent_state: Parent's CMA-ES state
        parent_layer_sizes: Parent's network architecture
        offspring_layer_sizes: Offspring's network architecture
        sigma_init: Initial sigma for new components
        covariance_mode: How to adapt covariance matrix
        sigma_mode: How to adapt step size
        rng: Random number generator (unused currently)

    Returns:
        Adapted CMAESState for offspring
    """
    # If architectures are identical, return parent state unchanged
    if parent_layer_sizes == offspring_layer_sizes:
        return parent_state

    # Calculate weight dimensions (including biases)
    parent_dim = _calculate_total_weights(parent_layer_sizes)
    offspring_dim = _calculate_total_weights(offspring_layer_sizes)

    # Adapt covariance matrix
    if covariance_mode == "reset":
        # No inheritance - start fresh
        offspring_covariance = np.eye(offspring_dim)

    elif covariance_mode == "preserve":
        # Experimental: Scale parent covariance to new size
        offspring_covariance = _scale_covariance_matrix(
            parent_state.covariance_matrix,
            parent_dim,
            offspring_dim,
        )

    elif covariance_mode == "adaptive":
        # Smart adaptation: preserve learned correlations where possible
        offspring_covariance = _adapt_covariance_intelligent(
            parent_state.covariance_matrix,
            parent_layer_sizes,
            offspring_layer_sizes,
        )

    else:
        raise ValueError(f"Unknown covariance_mode: {covariance_mode}")

    # Adapt sigma
    if sigma_mode == "reset":
        # No inheritance - always use initial sigma
        offspring_sigma = sigma_init

    elif sigma_mode == "keep":
        # Full inheritance - always use parent sigma
        offspring_sigma = parent_state.sigma

    elif sigma_mode in ["blend", "adaptive"]:
        # Blend based on proportion of new weights
        prop_new = abs(offspring_dim - parent_dim) / offspring_dim
        offspring_sigma = (1 - prop_new) * parent_state.sigma + prop_new * sigma_init

    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    # Adapt mean (same logic as covariance for now)
    if covariance_mode == "reset":
        offspring_mean = np.zeros(offspring_dim)
    else:
        offspring_mean = _adapt_mean_vector(
            parent_state.mean,
            parent_layer_sizes,
            offspring_layer_sizes,
        )

    # Create adapted state (no nevergrad state - can't be directly restored)
    adapted_state = CMAESState(
        nevergrad_state=None,  # Adapted states lose nevergrad state
        covariance_matrix=offspring_covariance,
        sigma=offspring_sigma,
        mean=offspring_mean,
        layer_sizes=offspring_layer_sizes,
    )

    # Compute condition number
    adapted_state.condition_number = adapted_state.compute_condition_number()

    return adapted_state


def _scale_covariance_matrix(
    parent_cov: np.ndarray,
    parent_dim: int,
    offspring_dim: int,
) -> np.ndarray:
    """Scale parent covariance matrix to new dimensionality (experimental).

    Simple approach: pad with identity or truncate.
    """
    offspring_cov = np.eye(offspring_dim)

    if offspring_dim >= parent_dim:
        # Offspring is larger: pad with identity
        offspring_cov[:parent_dim, :parent_dim] = parent_cov
    else:
        # Offspring is smaller: truncate
        offspring_cov = parent_cov[:offspring_dim, :offspring_dim].copy()

    return offspring_cov


def _adapt_covariance_intelligent(
    parent_cov: np.ndarray,
    parent_layer_sizes: list[int],
    offspring_layer_sizes: list[int],
) -> np.ndarray:
    """Intelligently adapt covariance matrix when architecture changes.

    This implements the strategy from lamarck_cmaes_strategy.txt:
    - Identify which weights are preserved vs new
    - Maintain learned correlations among preserved weights
    - Set new weights to identity (no learned correlations yet)

    Strategy:
    1. Build correspondence mapping between parent and offspring weights
    2. For weights that correspond (same semantic meaning), copy their covariances
    3. For new weights, use identity covariance (uncorrelated)

    This ensures that:
    - cov[w_i, w_j] is preserved when both w_i and w_j exist in parent and offspring
    - cov[w_new, w_anything] = 0 (except diagonal = 1)

    Example from lamarck_cmaes_strategy.txt:
    Parent weights [w0, w1, w2, w3] -> Child [w0, w1, w2_new, w3, w4, w5_new]
    - cov[w0, w1] preserved (both exist in parent and child)
    - cov[w3, w4] preserved (parent w2, w3 -> child w3, w4)
    - cov[w2_new, *] = 0 (new weight, no learned correlations)
    """
    # Calculate dimensions
    offspring_dim = _calculate_total_weights(offspring_layer_sizes)

    # Start with identity for offspring (all new weights uncorrelated)
    offspring_cov = np.eye(offspring_dim)

    # If same architecture, return parent covariance
    if parent_layer_sizes == offspring_layer_sizes:
        return parent_cov.copy()

    # Build weight correspondence mapping
    correspondences = _build_weight_correspondence(parent_layer_sizes, offspring_layer_sizes)

    # Copy covariances for all corresponding weight pairs
    # For each pair (i, j) of corresponding weights, copy cov[i, j]
    for p_i, o_i in correspondences:
        for p_j, o_j in correspondences:
            offspring_cov[o_i, o_j] = parent_cov[p_i, p_j]

    return offspring_cov


def _adapt_mean_vector(
    parent_mean: np.ndarray,
    parent_layer_sizes: list[int],
    offspring_layer_sizes: list[int],
) -> np.ndarray:
    """Adapt mean vector when architecture changes.

    Similar strategy to covariance: preserve mean for corresponding weights,
    zero for new weights.
    """
    offspring_dim = _calculate_total_weights(offspring_layer_sizes)

    # Start with zeros for offspring (new weights start at zero)
    offspring_mean = np.zeros(offspring_dim)

    # If same architecture, return parent mean
    if parent_layer_sizes == offspring_layer_sizes:
        return parent_mean.copy()

    # Build weight correspondence mapping
    correspondences = _build_weight_correspondence(parent_layer_sizes, offspring_layer_sizes)

    # Copy mean for all corresponding weights
    for p_i, o_i in correspondences:
        offspring_mean[o_i] = parent_mean[p_i]

    return offspring_mean


def _build_weight_correspondence(
    parent_layer_sizes: list[int],
    offspring_layer_sizes: list[int],
) -> list[tuple[int, int]]:
    """Build correspondence mapping between parent and offspring weight indices.

    Returns list of (parent_idx, offspring_idx) pairs for weights that have
    the same semantic meaning in both networks. This respects the structure
    of weight matrices and biases.

    Args:
        parent_layer_sizes: Parent network architecture
        offspring_layer_sizes: Offspring network architecture

    Returns:
        List of (parent_idx, offspring_idx) tuples for corresponding weights
    """
    correspondences = []
    parent_offset = 0
    offspring_offset = 0

    # Process each layer
    num_layers = min(len(parent_layer_sizes) - 1, len(offspring_layer_sizes) - 1)

    for layer_idx in range(num_layers):
        p_in = parent_layer_sizes[layer_idx]
        p_out = parent_layer_sizes[layer_idx + 1]
        o_in = offspring_layer_sizes[layer_idx]
        o_out = offspring_layer_sizes[layer_idx + 1]

        # Process weight matrix (flattened row-major)
        # Parent weight matrix: p_in × p_out
        # Offspring weight matrix: o_in × o_out
        # Overlapping region: min(p_in, o_in) × min(p_out, o_out)

        min_in = min(p_in, o_in)
        min_out = min(p_out, o_out)

        for i in range(min_in):
            for j in range(min_out):
                # Index in parent's flattened weight matrix
                parent_w_idx = parent_offset + i * p_out + j
                # Index in offspring's flattened weight matrix
                offspring_w_idx = offspring_offset + i * o_out + j
                correspondences.append((parent_w_idx, offspring_w_idx))

        # Move past weight matrices
        parent_weight_offset = parent_offset + p_in * p_out
        offspring_weight_offset = offspring_offset + o_in * o_out

        # Process biases (direct 1-to-1 correspondence for overlapping portion)
        min_biases = min(p_out, o_out)
        for b in range(min_biases):
            parent_b_idx = parent_weight_offset + b
            offspring_b_idx = offspring_weight_offset + b
            correspondences.append((parent_b_idx, offspring_b_idx))

        # Update offsets for next layer
        parent_offset = parent_weight_offset + p_out
        offspring_offset = offspring_weight_offset + o_out

    return correspondences


def _calculate_layer_weight_indices(layer_sizes: list[int]) -> list[int]:
    """Calculate start indices for each layer's weights in flattened vector.

    The weight vector is organized as:
    [layer0_weights, layer0_biases, layer1_weights, layer1_biases, ...]

    Returns:
        List of start indices, with final element being total dimension
    """
    indices = [0]
    cumsum = 0

    for i in range(len(layer_sizes) - 1):
        # Weights for this layer
        layer_weights = layer_sizes[i] * layer_sizes[i + 1]
        # Biases for this layer
        layer_biases = layer_sizes[i + 1]
        # Total for this layer
        cumsum += layer_weights + layer_biases
        indices.append(cumsum)

    return indices


class CMAESStateManager:
    """Manages CMA-ES state storage and retrieval for inheritance.

    Mirrors the ParentWeightManager pattern used for weight inheritance.
    """

    def __init__(
        self,
        sigma_init: float = 1.0,
        covariance_mode: Literal["adaptive", "reset", "preserve"] = "adaptive",
        sigma_mode: Literal["blend", "reset", "keep", "adaptive"] = "blend",
    ):
        """Initialize the CMA-ES state manager.

        Args:
            sigma_init: Initial sigma for new components
            covariance_mode: How to adapt covariance matrices
            sigma_mode: How to adapt step sizes
        """
        self.sigma_init = sigma_init
        self.covariance_mode = covariance_mode
        self.sigma_mode = sigma_mode

        # Storage: individual_id -> CMAESState
        self._states: dict[int, CMAESState] = {}

        # Track layer sizes for each individual
        self._layer_sizes: dict[int, list[int]] = {}

    def store_state(
        self,
        individual_id: int,
        state: CMAESState,
    ) -> None:
        """Store CMA-ES state for an individual.

        Args:
            individual_id: Unique identifier for the individual
            state: CMA-ES state to store
        """
        self._states[individual_id] = state
        self._layer_sizes[individual_id] = state.layer_sizes

    def get_state(
        self,
        individual_id: int,
    ) -> CMAESState | None:
        """Retrieve CMA-ES state for an individual.

        Args:
            individual_id: Unique identifier for the individual

        Returns:
            Stored CMA-ES state, or None if not found
        """
        return self._states.get(individual_id)

    def get_inherited_state(
        self,
        offspring: Individual,
        offspring_layer_sizes: list[int],
        parent_individuals: list[Individual],
        rng: Generator,
    ) -> CMAESState | None:
        """Get inherited CMA-ES state for an offspring.

        Uses closest parent strategy: inherit from the parent with the most
        similar morphology structure.

        Args:
            offspring: The offspring individual
            offspring_layer_sizes: Offspring's network architecture
            parent_individuals: List of parent individuals
            rng: Random number generator

        Returns:
            Adapted CMA-ES state, or None if no parents have states
        """
        if not parent_individuals:
            return None

        # Find closest parent by tree structure similarity
        closest_parent = self._find_closest_parent(offspring, parent_individuals)

        if closest_parent is None:
            return None

        # Get parent state
        parent_state = self.get_state(closest_parent.id)
        if parent_state is None:
            return None

        parent_layer_sizes = self._layer_sizes.get(closest_parent.id)
        if parent_layer_sizes is None:
            parent_layer_sizes = parent_state.layer_sizes

        # Adapt to offspring morphology
        adapted_state = adapt_cmaes_state_to_morphology(
            parent_state=parent_state,
            parent_layer_sizes=parent_layer_sizes,
            offspring_layer_sizes=offspring_layer_sizes,
            sigma_init=self.sigma_init,
            covariance_mode=self.covariance_mode,
            sigma_mode=self.sigma_mode,
            rng=rng,
        )

        return adapted_state

    def _find_closest_parent(
        self,
        offspring: Individual,
        parent_individuals: list[Individual],
    ) -> Individual | None:
        """Find the parent with the most similar tree structure.

        Uses tree_distance from weight_inheritance module.
        """
        offspring_genotype = offspring.genotype

        # Filter parents that have stored states
        parents_with_states = [
            parent for parent in parent_individuals
            if parent.id in self._states
        ]

        if not parents_with_states:
            return None

        # Calculate distances to all parents
        distances = [
            tree_distance(offspring_genotype, parent.genotype)
            for parent in parents_with_states
        ]

        # Return closest parent
        min_idx = int(np.argmin(distances))
        return parents_with_states[min_idx]

    def clear(self) -> None:
        """Clear all stored states (for memory management)."""
        self._states.clear()
        self._layer_sizes.clear()

    def __len__(self) -> int:
        """Return number of stored states."""
        return len(self._states)


def save_cmaes_state_to_disk(
    state: CMAESState,
    directory: Path,
) -> None:
    """Save CMA-ES state to individual's directory.

    Creates multiple files:
    - cmaes_state.pkl: Full pickled nevergrad state (if available)
    - cmaes_covariance.npy: Covariance matrix as NumPy array
    - cmaes_sigma.txt: Sigma value in plain text

    Args:
        state: CMA-ES state to save
        directory: Directory to save files in
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Save full nevergrad state if available
    if state.nevergrad_state is not None:
        with open(directory / "cmaes_state.pkl", 'wb') as f:
            pickle.dump(state.nevergrad_state, f)

    # Save covariance matrix
    np.save(directory / "cmaes_covariance.npy", state.covariance_matrix)

    # Save sigma as text (human-readable)
    with open(directory / "cmaes_sigma.txt", 'w') as f:
        f.write(f"{state.sigma}\n")

    # Save mean vector
    np.save(directory / "cmaes_mean.npy", state.mean)


def load_cmaes_state_from_disk(
    directory: Path,
    layer_sizes: list[int],
) -> CMAESState | None:
    """Load CMA-ES state from individual's directory.

    Args:
        directory: Directory containing saved files
        layer_sizes: Expected network architecture

    Returns:
        Loaded CMAESState, or None if files don't exist
    """
    directory = Path(directory)

    # Check if covariance file exists (minimum requirement)
    cov_file = directory / "cmaes_covariance.npy"
    if not cov_file.exists():
        return None

    # Load covariance matrix
    covariance = np.load(cov_file)

    # Load sigma
    sigma_file = directory / "cmaes_sigma.txt"
    if sigma_file.exists():
        with open(sigma_file, 'r') as f:
            sigma = float(f.read().strip())
    else:
        sigma = 1.0  # Default if not found

    # Load mean
    mean_file = directory / "cmaes_mean.npy"
    if mean_file.exists():
        mean = np.load(mean_file)
    else:
        mean = np.zeros(covariance.shape[0])

    # Load nevergrad state if available
    nevergrad_state = None
    state_file = directory / "cmaes_state.pkl"
    if state_file.exists():
        with open(state_file, 'rb') as f:
            nevergrad_state = pickle.load(f)

    # Create state object
    state = CMAESState(
        nevergrad_state=nevergrad_state,
        covariance_matrix=covariance,
        sigma=sigma,
        mean=mean,
        layer_sizes=layer_sizes,
    )

    # Compute condition number
    state.condition_number = state.compute_condition_number()

    return state
