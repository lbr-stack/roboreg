from abc import ABC, abstractmethod

import numpy as np
import rich
import torch


class ParticleSwarmOptimizer(ABC):
    __slots__ = [
        "_device",
        "_random_particles",
        "_best_particles",
        "_best_particle",
        "_best_particles_fitness",
        "_best_fitness",
        "_velocity",
    ]

    def __init__(self, device: torch.device="cuda") -> None:
        self._device = device
        self._random_particles = self._init_random_particles()
        if not isinstance(self._random_particles, torch.Tensor):
            raise ValueError(
                f"Expected _init_random_particles to return a torch.Tensor, got {type(self._random_particles)}"
            )
        if self._random_particles.device != self._device:
            raise ValueError(
                f"Expected _init_random_particles to return a tensor on device {self._device}, got {self._random_particles.device}"
            )
        self._best_particles = None
        self._best_particle = None
        self._best_particles_fitness = None
        self._best_fitness = None
        self._velocity = None
        self._init_state()

    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def best_particle(self) -> torch.Tensor:
        return self._best_particle
    
    @property
    def best_fitness(self) -> float:
        return self._best_fitness.item()

    @abstractmethod
    def _fitness_function(self, particles: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _init_random_particles(self) -> torch.Tensor:
        raise NotImplementedError

    def _init_state(self) -> None:
        self._best_particles = self._random_particles.clone()  # B x DoF
        self._best_particle = torch.zeros_like(self._random_particles[0])  # DoF
        self._best_particles_fitness = torch.full_like(
            self._random_particles[:, 0], float("inf")
        )  # B
        self._best_fitness = float("inf")
        self._velocity = torch.zeros_like(self._random_particles)

    @torch.no_grad()
    def __call__(
        self,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iterations: int = 100,
        min_velocity: float = 1.0e-4,
    ) -> torch.Tensor:
        try:
            iteration = 0
            while iteration < max_iterations and self._velocity.mean() > min_velocity:
                iteration += 1

                # compute particle veloctiy
                r1 = torch.rand_like(self._random_particles)
                r2 = torch.rand_like(self._random_particles)

                # update particle velocity
                self._velocity = (
                    w * self._velocity
                    + c1 * r1 * (self._best_particles - self._random_particles)
                    + c2 * r2 * (self._best_particle - self._random_particles)
                )

                # update particle position
                self._random_particles = self._random_particles + self._velocity

                # evaluate fitness
                fitnesses = self._fitness_function(self._random_particles)

                # update particles best known losses and positions
                mask = fitnesses < self._best_particles_fitness
                self._best_particles_fitness[mask] = fitnesses[mask]
                self._best_particles[mask] = self._random_particles[mask]

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
        except KeyboardInterrupt:
            pass
        return self._best_particle
