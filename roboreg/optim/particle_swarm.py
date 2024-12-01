from abc import ABC, abstractmethod

import numpy as np
import rich
import torch
from rich import progress


class ParticleSwarm(ABC):
    __slots__ = ["_particles", "_particle_box_bounds", "_velocity"]

    def __init__(
        self, particles: torch.Tensor, particle_box_bounds: torch.Tensor
    ) -> None:
        if particles.device != particle_box_bounds.device:
            raise ValueError(
                "Particles and particle bounds must be on the same device."
            )
        if particle_box_bounds.shape[-1] != 2 or particle_box_bounds.ndim != 2:
            raise ValueError("Particle bounds must have shape (DoF, 2).")
        if particle_box_bounds[:, 0].ge(particle_box_bounds[:, 1]).any():
            raise ValueError("Lower bounds must be less than upper bounds.")
        if particles.shape[-1] != particle_box_bounds.shape[0]:
            raise ValueError(
                "Particles and particle bounds must have the same number of DoF."
            )
        if particles.ndim != 2:
            raise ValueError("Particles must be 2 dimensional.")

        self._particles = particles
        self._particle_box_bounds = particle_box_bounds
        self._velocity = torch.zeros_like(particles)

    @property
    def particles(self) -> torch.Tensor:
        return self._particles

    @property
    def particle_bounds(self) -> torch.Tensor:
        return self._particle_box_bounds

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @abstractmethod
    def compute_velocity(self) -> None:
        pass

    def step(self) -> None:
        self._particles += self._velocity
        self._particles = torch.clamp(  # note that these are simple box constraints
            self._particles,
            min=self._particle_box_bounds[:, 0],
            max=self._particle_box_bounds[:, 1],
        )


class LinearParticleSwarm(ParticleSwarm):
    __slots__ = ["_w", "_c1", "_c2"]

    def __init__(
        self,
        particles: torch.Tensor,
        particle_bounds: torch.Tensor,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> None:
        super().__init__(particles, particle_bounds)
        self._w = w
        self._c1 = c1
        self._c2 = c2

    def compute_velocity(
        self,
        best_particles: torch.Tensor,
        best_particle: torch.Tensor,
    ) -> None:
        r1 = torch.rand_like(self._particles)
        r2 = torch.rand_like(self._particles)
        self._velocity = (
            self._w * self._velocity
            + self._c1 * r1 * (best_particles - self._particles)
            + self._c2 * r2 * (best_particle - self._particles)
        )


class ParticleSwarmOptimizer:
    __slots__ = [
        "_particle_swarm",
        "_best_particles",
        "_best_particle",
        "_best_particles_fitness",
        "_best_fitness",
    ]

    def __init__(self, particle_swarm: ParticleSwarm) -> None:
        self._particle_swarm = particle_swarm
        self._best_particles = None
        self._best_particle = None
        self._best_particles_fitness = None
        self._best_fitness = None
        self._init_state()

    @property
    def particle_swarm(self) -> ParticleSwarm:
        return self._particle_swarm

    @property
    def best_particle(self) -> torch.Tensor:
        return self._best_particle

    @property
    def best_fitness(self) -> float:
        return self._best_fitness.item()

    def _init_state(self) -> None:
        self._best_particles = self._particle_swarm.particles.clone()  # B x DoF
        self._best_particle = torch.zeros_like(self._particle_swarm.particles[0])  # DoF
        self._best_particles_fitness = torch.full_like(
            self._particle_swarm.particles[:, 0], float("inf")
        )  # B
        self._best_fitness = float("inf")

    def __call__(
        self,
        fitness_function: callable,
        max_iterations: int = 100,
        min_velocity: float = 1.0e-4,
    ) -> torch.Tensor:
        try:
            with torch.no_grad():
                for iteration in progress.track(
                    range(max_iterations), description="Optimizing particle swarm..."
                ):
                    # update particle velocity
                    self._particle_swarm.compute_velocity(
                        self._best_particles,
                        self._best_particle,
                    )
                    self._particle_swarm.step()

                    # evaluate fitness
                    fitnesses = fitness_function()
                    if fitnesses.ndim != 1:
                        raise ValueError(
                            "Expected fitness function to return 1D tensor."
                        )
                    if fitnesses.shape != self._particle_swarm.particles.shape[0]:
                        raise ValueError(
                            "Expected fitness function to return fitness for each particle."
                        )

                    # update particles best known losses and positions
                    mask = fitnesses < self._best_particles_fitness
                    self._best_particles_fitness[mask] = fitnesses[mask]
                    self._best_particles[mask] = self._particle_swarm.particles[mask]

                    # update global best
                    best_fitness_idx = torch.argmin(self._best_particles_fitness)
                    best_fitness = self._best_particles_fitness[best_fitness_idx]
                    if best_fitness < self._best_fitness:
                        self._best_fitness = best_fitness
                        self._best_particle = self._best_particles[best_fitness_idx]
                        best_loss_print = np.round(self._best_fitness.item(), 3)
                        rich.print(
                            f"New best loss: {best_loss_print} at iteration {iteration}."
                        )
                    if self._particle_swarm.velocity.norm(dim=-1).mean() < min_velocity:
                        rich.print(
                            f"Velocity is below threshold {min_velocity}. Stopping optimization."
                        )
                        break

        except KeyboardInterrupt:
            pass
        return self._best_particle
