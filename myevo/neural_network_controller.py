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

        Input consists of:
        - Actuator positions (nu)
        - Actuator velocities (nu)
        - Core body orientation (3 Euler angles)
        - Core body linear velocity (3D)
        - Core body angular velocity (3D)
        Total: 2*nu + 9

        Parameters
        ----------
        model : mj.MjModel
            The MuJoCo model

        Returns
        -------
        int
            Total input size
        """
        # Input: actuator state (pos + vel) + core body state (orientation + vel + angvel)
        self.input_size = 2 * self.num_actuators + 9

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

        Constructs proprioceptive input from:
        - Actuator joint positions (nu values)
        - Actuator joint velocities (nu values)
        - Core body orientation (3 Euler angles from quaternion)
        - Core body linear velocity (3D)
        - Core body angular velocity (3D)
        Total: 2*nu + 9

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

        # Extract actuator joint states using model.actuator_trnid mapping
        actuator_qpos = np.zeros(self.num_actuators)
        actuator_qvel = np.zeros(self.num_actuators)

        for i in range(self.num_actuators):
            # Get the joint ID that this actuator controls
            joint_id = model.actuator_trnid[i, 0]
            if joint_id >= 0:
                # Get position and velocity addresses for this joint
                qposadr = model.jnt_qposadr[joint_id]
                dofadr = model.jnt_dofadr[joint_id]
                actuator_qpos[i] = data.qpos[qposadr]
                actuator_qvel[i] = data.qvel[dofadr]

        # Extract core body state from the free joint (always joint 0)
        # Free joint format: qpos[0:7] = [x, y, z, qw, qx, qy, qz]
        #                   qvel[0:6] = [vx, vy, vz, wx, wy, wz]
        if model.nq >= 7 and model.nv >= 6:
            # Extract quaternion and convert to Euler angles
            quat = data.qpos[3:7]  # [qw, qx, qy, qz]
            euler = self._quat_to_euler(quat)

            # Extract linear velocity
            linear_vel = data.qvel[0:3]

            # Extract angular velocity
            angular_vel = data.qvel[3:6]
        else:
            # Fallback if no free joint (shouldn't happen in our setup)
            euler = np.zeros(3)
            linear_vel = np.zeros(3)
            angular_vel = np.zeros(3)

        # Construct input vector: [actuator_pos, actuator_vel, orientation, lin_vel, ang_vel]
        x = np.concatenate([
            actuator_qpos,
            actuator_qvel,
            euler,
            linear_vel,
            angular_vel,
        ])

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

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw).

        Parameters
        ----------
        quat : np.ndarray
            Quaternion [qw, qx, qy, qz]

        Returns
        -------
        np.ndarray
            Euler angles [roll, pitch, yaw] in radians
        """
        qw, qx, qy, qz = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
