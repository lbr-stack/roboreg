from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import rich
import torch
from rich import progress


class ParticleSwarm(ABC):
    __slots__ = ["_particles", "_velocity"]

    def __init__(self, particles: torch.Tensor) -> None:
        if particles.ndim != 2:
            raise ValueError("Particles must be 2 dimensional.")

        self._particles = particles
        self._velocity = torch.zeros_like(particles)

    @property
    def particles(self) -> torch.Tensor:
        return self._particles

    @particles.setter
    def particles(self, particles: torch.Tensor) -> None:
        self._particles = particles

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @abstractmethod
    def compute_velocity(self) -> None:
        pass

    def reset(self) -> None:
        self._velocity = torch.zeros_like(self._particles)

    def step(self) -> None:
        self._particles += (
            self._velocity
        )  # note that one may potentially constrain particles here


class LinearParticleSwarm(ParticleSwarm):
    __slots__ = ["_w", "_c1", "_c2"]

    def __init__(
        self,
        particles: torch.Tensor,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> None:
        super().__init__(particles)
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
        return self._best_fitness

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
        min_fitness_change: float = 2.0e-3,
        max_iterations_below_min_fitness_change: int = 20,
    ) -> Tuple[torch.Tensor, float]:
        try:
            prev_best_fitness = self._best_fitness
            iterations_below_min_fitness_change = 0
            with torch.no_grad():
                for iteration in progress.track(
                    range(1, max_iterations + 1),
                    description="Optimizing particle swarm...",
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
                    if fitnesses.shape[0] != self._particle_swarm.particles.shape[0]:
                        raise ValueError(
                            "Expected fitness function to return fitness for each particle."
                        )

                    # update particles best known losses and positions
                    mask = fitnesses < self._best_particles_fitness
                    self._best_particles_fitness[mask] = fitnesses[mask]
                    self._best_particles[mask] = self._particle_swarm.particles[mask]

                    # update global best
                    best_fitness_idx = torch.argmin(self._best_particles_fitness)
                    best_fitness = self._best_particles_fitness[best_fitness_idx].item()
                    if best_fitness < self._best_fitness:
                        self._best_fitness = best_fitness
                        self._best_particle = self._best_particles[best_fitness_idx]
                        best_loss_print = np.round(self._best_fitness, 3)
                        rich.print(
                            f"Step [{iteration} / {max_iterations}], new best loss: {best_loss_print} at iteration {iteration}."
                        )
                    if abs(prev_best_fitness - self._best_fitness) < min_fitness_change:
                        iterations_below_min_fitness_change += 1
                    else:
                        iterations_below_min_fitness_change = 0
                    prev_best_fitness = self._best_fitness
                    if (
                        iterations_below_min_fitness_change
                        >= max_iterations_below_min_fitness_change
                    ):
                        rich.print(
                            f"Stopping optimization as best fitness has not changed more than {min_fitness_change} for {max_iterations_below_min_fitness_change} iterations."
                        )
                        break

        except KeyboardInterrupt:
            pass
        return self._best_particle, self._best_fitness
