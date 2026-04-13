import numpy as np

# --------------------------------------------------------------------------
# Network architecture:  5 → 8 → 8 → 1   (tiny MLP, no PyTorch needed)
# Output > 0  →  flap,   output <= 0  →  do nothing
# --------------------------------------------------------------------------

LAYER_SIZES = [5, 8, 8, 1]


def _init_weights():
    """Return a list of (W, b) tuples for each layer transition."""
    layers = []
    for i in range(len(LAYER_SIZES) - 1):
        fan_in  = LAYER_SIZES[i]
        fan_out = LAYER_SIZES[i + 1]
        W = np.random.randn(fan_out, fan_in).astype(np.float32) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out, dtype=np.float32)
        layers.append((W, b))
    return layers


def forward(layers, x):
    """Run a forward pass. x: shape (5,). Returns scalar."""
    for i, (W, b) in enumerate(layers):
        x = W @ x + b
        if i < len(layers) - 1:
            x = np.tanh(x)           # hidden activation
    return float(x[0])               # raw output, no final activation


def decide(layers, obs):
    """1 = flap, 0 = do nothing."""
    return 1 if forward(layers, obs) > 0.0 else 0


# --------------------------------------------------------------------------
# Genome helpers
# --------------------------------------------------------------------------

def flatten(layers):
    """Flatten all weights into a single 1-D array."""
    return np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in layers])


def unflatten(genome):
    """Reconstruct layers from a flat genome array."""
    layers = []
    idx = 0
    for i in range(len(LAYER_SIZES) - 1):
        fan_in  = LAYER_SIZES[i]
        fan_out = LAYER_SIZES[i + 1]
        n_W = fan_out * fan_in
        n_b = fan_out
        W = genome[idx: idx + n_W].reshape(fan_out, fan_in)
        b = genome[idx + n_W: idx + n_W + n_b]
        layers.append((W.astype(np.float32), b.astype(np.float32)))
        idx += n_W + n_b
    return layers


def genome_size():
    total = 0
    for i in range(len(LAYER_SIZES) - 1):
        total += LAYER_SIZES[i] * LAYER_SIZES[i + 1] + LAYER_SIZES[i + 1]
    return total


# --------------------------------------------------------------------------
# Genetic algorithm
# --------------------------------------------------------------------------

class GeneticTrainer:
    def __init__(self, n_birds=8, mutation_rate=0.15, mutation_scale=0.3, elite=2):
        self.n_birds       = n_birds
        self.mutation_rate = mutation_rate   # probability each weight is perturbed
        self.mutation_scale= mutation_scale  # std of perturbation
        self.elite         = elite           # how many top genomes survive unchanged

        g_size = genome_size()
        # initialise population as flat genomes
        self.population = [
            np.random.randn(g_size).astype(np.float32) * 0.5
            for _ in range(n_birds)
        ]
        self.generation  = 0
        self.best_genome = None
        self.best_fitness= -np.inf
        self.best_score  = 0

    # ------------------------------------------------------------------
    def networks(self):
        """Return list of layer-lists (one per bird)."""
        return [unflatten(g) for g in self.population]

    # ------------------------------------------------------------------
    def evolve(self, fitnesses, scores):
        """
        fitnesses: list of float, one per bird (steps alive + pipe bonus)
        scores:    list of int pipe counts
        """
        ranked = sorted(range(self.n_birds), key=lambda i: fitnesses[i], reverse=True)
        best_i = ranked[0]

        if fitnesses[best_i] > self.best_fitness:
            self.best_fitness = fitnesses[best_i]
            self.best_genome  = self.population[best_i].copy()
            self.best_score   = scores[best_i]

        # always keep the all-time best in the pool
        elites = [self.population[i].copy() for i in ranked[:self.elite]]
        if self.best_genome is not None and not np.array_equal(elites[0], self.best_genome):
            elites[0] = self.best_genome.copy()

        new_pop = elites[:]

        # fill rest by mutating parents (fitness-proportional selection)
        fs = np.array(fitnesses, dtype=np.float64)
        fs = np.clip(fs - fs.min(), 0, None)
        total = fs.sum()
        probs = fs / total if total > 0 else np.ones(self.n_birds) / self.n_birds

        while len(new_pop) < self.n_birds:
            parent_i = np.random.choice(self.n_birds, p=probs)
            child    = self._mutate(self.population[parent_i])
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1

        return ranked[0], fitnesses[best_i], scores[best_i]

    # ------------------------------------------------------------------
    def _mutate(self, genome):
        child = genome.copy()
        mask  = np.random.rand(len(child)) < self.mutation_rate
        child[mask] += np.random.randn(mask.sum()).astype(np.float32) * self.mutation_scale
        return child

    # ------------------------------------------------------------------
    def save(self, path="evo_checkpoint.npz"):
        np.savez(path,
                 population  = np.stack(self.population),
                 best_genome = self.best_genome if self.best_genome is not None
                               else np.zeros(genome_size()),
                 generation  = np.array([self.generation]),
                 best_fitness= np.array([self.best_fitness]),
                 best_score  = np.array([self.best_score]))
        print(f"  saved — gen {self.generation} | best score {self.best_score}")

    def load(self, path="evo_checkpoint.npz"):
        data = np.load(path)
        self.population   = list(data["population"])
        self.best_genome  = data["best_genome"]
        self.generation   = int(data["generation"][0])
        self.best_fitness = float(data["best_fitness"][0])
        self.best_score   = int(data["best_score"][0])
        print(f"  loaded — gen {self.generation} | best score {self.best_score}")