"""Flexible neural network controller for robot locomotion.

This module provides a configurable neural network controller that can be used
with various robot morphologies and supports different architectures and
activation functions.
"""

import mujoco as mj
import numpy as np
import numpy.typing as npt


class FlexibleNeuralNetworkController:
    """Flexible neural network controller with configurable architecture.

    Supports variable number of hidden layers, neurons per layer, and
    activation functions.
    """

    ACTIVATION_FUNCTIONS = {
        'tanh': np.tanh,
        'relu': lambda x: np.maximum(0, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
    }

    def __init__(
        self,
        num_actuators: int,
        hidden_layers: list[int],
        activation: str = 'tanh',
        seed: int = 42,
    ):
        """Initialize the flexible neural network controller.

        Parameters
        ----------
        num_actuators : int
            Number of actuators in the robot
        hidden_layers : list[int]
            List of hidden layer sizes (e.g., [32, 16, 32] for three hidden layers)
        activation : str
            Activation function name ('tanh', 'relu', 'sigmoid', 'elu')
        seed : int
            Random seed for initialization
        """
        self.num_actuators = num_actuators
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.activation_fn = self.ACTIVATION_FUNCTIONS[activation]
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Will be set when we know the input size
        self.input_size = None
        self.weights = None
        self.layer_sizes = None

    def calculate_input_size(self, model: mj.MjModel) -> int:
        """Calculate the input size based on the model.

        Parameters
        ----------
        model : mj.MjModel
            The MuJoCo model

        Returns
        -------
        int
            Total input size
        """
        self.input_size = model.nq + model.nv

        # Define layer sizes: [input] + hidden_layers + [output]
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.num_actuators]

        return self.input_size

    def get_num_weights(self) -> int:
        """Get the total number of weights in the network.

        Returns
        -------
        int
            Total number of weights and biases
        """
        if self.layer_sizes is None:
            msg = "Input size not set. Call calculate_input_size first."
            raise ValueError(msg)

        total = 0
        for i in range(len(self.layer_sizes) - 1):
            # Weights: layer_sizes[i] * layer_sizes[i+1]
            # Biases: layer_sizes[i+1]
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]
            total += self.layer_sizes[i + 1]

        return total

    def set_weights(self, flat_weights: np.ndarray) -> None:
        """Set the network weights from a flat array.

        Parameters
        ----------
        flat_weights : np.ndarray
            Flattened array of all weights and biases
        """
        if self.layer_sizes is None:
            msg = "Input size not set. Call calculate_input_size first."
            raise ValueError(msg)

        expected_size = self.get_num_weights()
        if len(flat_weights) != expected_size:
            msg = f"Expected {expected_size} weights, got {len(flat_weights)}"
            raise ValueError(msg)

        # Extract weights and biases for each layer
        self.weights = []
        idx = 0

        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]

            # Extract weight matrix
            w_size = in_size * out_size
            w = flat_weights[idx:idx + w_size].reshape(in_size, out_size)
            idx += w_size

            # Extract bias vector
            b = flat_weights[idx:idx + out_size]
            idx += out_size

            self.weights.append({'w': w, 'b': b})

    def __call__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
    ) -> npt.NDArray[np.float64]:
        """Execute the neural network controller.

        Parameters
        ----------
        model : mj.MjModel
            The MuJoCo model
        data : mj.MjData
            The MuJoCo data

        Returns
        -------
        npt.NDArray[np.float64]
            Joint angle commands
        """
        if self.weights is None:
            msg = "Weights not set. Call set_weights first."
            raise ValueError(msg)

        # Construct input vector
        x = np.concatenate([data.qpos, data.qvel])

        # Forward pass through all layers
        for i, layer in enumerate(self.weights):
            x = np.dot(x, layer['w']) + layer['b']

            # Apply activation (use tanh on output layer for bounded control)
            if i < len(self.weights) - 1:
                x = self.activation_fn(x)
            else:
                x = np.tanh(x)

        # Scale outputs to joint angle range
        return x * (np.pi / 2)
