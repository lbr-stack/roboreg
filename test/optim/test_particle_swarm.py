import os
import sys

sys.path.append(
    os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import torch

from roboreg.optim import LinearParticleSwarm, ParticleSwarmOptimizer


def test_particle_swarm() -> None:
    n_particles = 10
    particle_dof = 7

    particles = torch.rand(n_particles, particle_dof)
    best_particles = torch.rand(n_particles, particle_dof)
    best_particle = torch.rand(particle_dof)
    swarm = LinearParticleSwarm(
        particles=particles,
    )
    swarm.compute_velocity(
        best_particles=best_particles,
        best_particle=best_particle,
    )
    swarm.step()

    # test invalid particle dimensions
    particles = torch.rand(1, 1, 1)
    try:
        swarm = LinearParticleSwarm(
            particles=particles,
        )
    except ValueError as e:
        print(f"Expected and got error: {e}")
    else:
        raise ValueError("Expected ValueError")


def test_particle_swarm_optimizer() -> None:
    n_particles = 10
    particle_dof = 2

    bound = 500.0
    particles = torch.rand(n_particles, particle_dof) * 2 * bound - bound
    particle_swarm = LinearParticleSwarm(
        particles=particles,
    )
    pso = ParticleSwarmOptimizer(
        particle_swarm=particle_swarm,
    )

    capture_value = 2

    def fitness_closure() -> torch.Tensor:
        if capture_value != 2:
            raise ValueError("Variable wasn't captured.")
        return torch.square(pso.particle_swarm.particles).sum(dim=1)

    best_particle = pso(
        fitness_function=fitness_closure, max_iterations=100, min_velocity=0.1
    )

    if not torch.isclose(best_particle, torch.zeros(particle_dof), atol=1.0e-3).all():
        raise ValueError("Expected best particle to be [0, 0].")


if __name__ == "__main__":
    test_particle_swarm()
    test_particle_swarm_optimizer()
