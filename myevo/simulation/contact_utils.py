"""Utilities for tracking ground contact events during robot simulation.

This module provides functions to detect when hinge rotors make contact with
the ground, which is useful for penalizing morphologies that exploit contact
glitches for locomotion.
"""

import mujoco as mj


def get_rotors_in_contact(model: mj.MjModel, data: mj.MjData) -> set[str]:
    """Find which rotor geoms are currently in contact with the ground.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model.
    data : mj.MjData
        The MuJoCo simulation data.

    Returns
    -------
    set[str]
        Set of rotor geom names currently in contact with floor.
    """
    rotors_touching = set()
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, contact.geom2)

        # Check if one geom is a rotor and the other is the floor
        if geom1_name and geom2_name:
            if "rotor" in geom1_name and "floor" in geom2_name:
                rotors_touching.add(geom1_name)
            elif "rotor" in geom2_name and "floor" in geom1_name:
                rotors_touching.add(geom2_name)

    return rotors_touching


def count_contact_events_during_simulation(
    model: mj.MjModel,
    data: mj.MjData,
    num_steps: int,
) -> int:
    """Count unique contact events during simulation steps.

    A contact event occurs when a rotor that was not touching the ground
    in the previous step begins touching the ground in the current step.
    This counts transitions from "not in contact" to "in contact".

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model.
    data : mj.MjData
        The MuJoCo simulation data (will be modified by stepping).
    num_steps : int
        Number of simulation steps to run and monitor.

    Returns
    -------
    int
        Total number of unique contact events detected.
    """
    previous_rotors_in_contact = set()
    unique_contact_events = 0

    for step in range(num_steps):
        # Step the simulation
        mj.mj_step(model, data)

        # Check current contacts
        current_rotors_in_contact = get_rotors_in_contact(model, data)

        # Count new contacts (rotors that are touching now but weren't before)
        new_contacts = current_rotors_in_contact - previous_rotors_in_contact
        unique_contact_events += len(new_contacts)

        # Update tracking set for next iteration
        previous_rotors_in_contact = current_rotors_in_contact

    return unique_contact_events
