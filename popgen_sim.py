"""
Population Genetics Simulation Module (Vectorized)

Fast NumPy-based simulation for exploring allele frequencies, 
genotype frequencies, and linkage disequilibrium dynamics.

Supports:
- 1 or 2 locus models
- Random, assortative, and disassortative mating
- Variable population sizes, recombination rates
- Multiple replicates for drift visualization

Nomenclature:
- p_A, p_B: allele frequencies
- G_AA, G_Aa, G_aa: genotype frequencies (capital G)
- g_AB, g_Ab, g_aB, g_ab: gamete/haplotype frequencies (lowercase g)
- D, r²: linkage disequilibrium measures
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


# =============================================================================
# Color Schemes
# =============================================================================

# Locus A: Blues (light to dark for aa, Aa, AA)
COLORS_A = {
    'aa': '#a6cee3',  # light blue
    'Aa': '#1f78b4',  # medium blue
    'AA': '#08306b',  # dark blue
}

# Locus B: Oranges (light to dark for bb, Bb, BB)
COLORS_B = {
    'bb': '#fdbf6f',  # light orange
    'Bb': '#ff7f00',  # medium orange
    'BB': '#b15928',  # dark orange
}

# Gametes: blue-orange gradient
COLORS_G = {
    'AB': '#08306b',  # dark blue (both dominant)
    'Ab': '#1f78b4',  # medium blue (A dom, b rec)
    'aB': '#ff7f00',  # medium orange (a rec, B dom)
    'ab': '#fdbf6f',  # light orange (both recessive)
}

# Allele line colors
COLOR_A = '#1f78b4'  # blue
COLOR_B = '#ff7f00'  # orange


# =============================================================================
# Enums and Parameters
# =============================================================================

class MatingSystem(Enum):
    RANDOM = 'random'
    ASSORTATIVE = 'assortative'
    DISASSORTATIVE = 'disassortative'


@dataclass
class SimParams:
    """Simulation parameters with Hardy-Weinberg defaults."""
    n_loci: int = 2
    pop_size: int = 1000
    n_generations: int = 50
    n_replicates: int = 1
    
    freq_A: float = 0.5
    freq_B: float = 0.5
    initial_D: float = 0.0
    
    mating_system: MatingSystem = MatingSystem.RANDOM
    assortment_strength: float = 1.0
    assortment_trait: str = 'additive'
    
    recomb_rate: float = 0.5
    
    fitness: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if isinstance(self.mating_system, str):
            self.mating_system = MatingSystem(self.mating_system)
        if self.n_loci not in [1, 2]:
            raise ValueError("n_loci must be 1 or 2")
    
    def get_D_bounds(self) -> Tuple[float, float]:
        """Valid D range for current allele frequencies."""
        if self.n_loci == 1:
            return (0.0, 0.0)
        p_A, p_B = self.freq_A, self.freq_B
        p_a, p_b = 1 - p_A, 1 - p_B
        D_min = max(-p_A * p_B, -p_a * p_b)
        D_max = min(p_A * p_b, p_a * p_B)
        return (D_min, D_max)


# =============================================================================
# Vectorized Population Class
# =============================================================================

class Population:
    """
    Vectorized population using NumPy arrays.
    
    Internal representation:
    - haplotypes: array of shape (N, 2, n_loci)
      - haplotypes[i, j, k] = allele at locus k on haplotype j of individual i
      - Allele coding: 1 = A/B (dominant), 0 = a/b (recessive)
    """
    
    def __init__(self, haplotypes: np.ndarray):
        self.haplotypes = haplotypes
        self.n = haplotypes.shape[0]
        self.n_loci = haplotypes.shape[2]
    
    @classmethod
    def from_params(cls, params: SimParams) -> 'Population':
        """Initialize population from SimParams."""
        n = params.pop_size
        n_loci = params.n_loci
        
        if n_loci == 1:
            p_A = params.freq_A
            haplotypes = (np.random.random((n, 2, 1)) < p_A).astype(np.int8)
        else:
            p_A, p_B, D = params.freq_A, params.freq_B, params.initial_D
            p_a, p_b = 1 - p_A, 1 - p_B
            
            # Gamete frequencies
            g_AB = np.clip(p_A * p_B + D, 0, 1)
            g_Ab = np.clip(p_A * p_b - D, 0, 1)
            g_aB = np.clip(p_a * p_B - D, 0, 1)
            g_ab = np.clip(p_a * p_b + D, 0, 1)
            
            total = g_AB + g_Ab + g_aB + g_ab
            freqs = np.array([g_AB, g_Ab, g_aB, g_ab]) / total
            
            # Gamete templates: AB=0, Ab=1, aB=2, ab=3
            templates = np.array([
                [1, 1],  # AB
                [1, 0],  # Ab
                [0, 1],  # aB
                [0, 0],  # ab
            ], dtype=np.int8)
            
            gamete_indices = np.random.choice(4, size=(n, 2), p=freqs)
            haplotypes = templates[gamete_indices]
        
        return cls(haplotypes)
    
    def allele_freq_A(self) -> float:
        return self.haplotypes[:, :, 0].mean()
    
    def allele_freq_B(self) -> float:
        if self.n_loci == 1:
            return 0.0
        return self.haplotypes[:, :, 1].mean()
    
    def genotypes_A(self) -> np.ndarray:
        return self.haplotypes[:, :, 0].sum(axis=1)
    
    def genotypes_B(self) -> np.ndarray:
        if self.n_loci == 1:
            return np.zeros(self.n, dtype=np.int8)
        return self.haplotypes[:, :, 1].sum(axis=1)
    
    def genotype_freqs_A(self) -> Dict[str, float]:
        g = self.genotypes_A()
        return {
            'aa': (g == 0).mean(),
            'Aa': (g == 1).mean(),
            'AA': (g == 2).mean(),
        }
    
    def genotype_freqs_B(self) -> Dict[str, float]:
        if self.n_loci == 1:
            return {}
        g = self.genotypes_B()
        return {
            'bb': (g == 0).mean(),
            'Bb': (g == 1).mean(),
            'BB': (g == 2).mean(),
        }
    
    def gamete_freqs(self) -> Dict[str, float]:
        if self.n_loci == 1:
            p = self.allele_freq_A()
            return {'A': p, 'a': 1 - p}
        
        haps = self.haplotypes.reshape(-1, 2)
        n_total = len(haps)
        
        return {
            'AB': ((haps[:, 0] == 1) & (haps[:, 1] == 1)).sum() / n_total,
            'Ab': ((haps[:, 0] == 1) & (haps[:, 1] == 0)).sum() / n_total,
            'aB': ((haps[:, 0] == 0) & (haps[:, 1] == 1)).sum() / n_total,
            'ab': ((haps[:, 0] == 0) & (haps[:, 1] == 0)).sum() / n_total,
        }
    
    def D(self) -> float:
        if self.n_loci == 1:
            return 0.0
        g = self.gamete_freqs()
        p_A = g['AB'] + g['Ab']
        p_B = g['AB'] + g['aB']
        return g['AB'] - p_A * p_B
    
    def r_squared(self) -> float:
        if self.n_loci == 1:
            return 0.0
        d = self.D()
        p_A = self.allele_freq_A()
        p_B = self.allele_freq_B()
        denom = p_A * (1 - p_A) * p_B * (1 - p_B)
        return (d ** 2 / denom) if denom > 1e-10 else 0.0
    
    def phenotypes(self, trait: str = 'additive') -> np.ndarray:
        if trait == 'additive':
            return self.genotypes_A() + self.genotypes_B()
        elif trait == 'locus_a':
            return self.genotypes_A()
        elif trait == 'locus_b':
            return self.genotypes_B()
        return self.genotypes_A()
    
    def stats(self) -> Dict:
        result = {
            'p_A': self.allele_freq_A(),
            'G_A': self.genotype_freqs_A(),
        }
        if self.n_loci == 2:
            result.update({
                'p_B': self.allele_freq_B(),
                'G_B': self.genotype_freqs_B(),
                'g': self.gamete_freqs(),
                'D': self.D(),
                'r_squared': self.r_squared(),
            })
        return result


# =============================================================================
# Vectorized Mating Functions
# =============================================================================

def _produce_gametes(haplotypes: np.ndarray, recomb_rate: float) -> np.ndarray:
    """Produce gametes from parents (vectorized)."""
    n, _, n_loci = haplotypes.shape
    
    if n_loci == 1:
        which = np.random.randint(0, 2, size=n)
        return haplotypes[np.arange(n), which, :]
    
    start_hap = np.random.randint(0, 2, size=n)
    recomb = np.random.random(n) < recomb_rate
    
    source_0 = start_hap
    source_1 = np.where(recomb, 1 - start_hap, start_hap)
    
    gametes = np.empty((n, n_loci), dtype=np.int8)
    gametes[:, 0] = haplotypes[np.arange(n), source_0, 0]
    gametes[:, 1] = haplotypes[np.arange(n), source_1, 1]
    
    return gametes


def mate_random(pop: Population, params: SimParams) -> Population:
    """Random mating (vectorized)."""
    n = pop.n
    
    parent1_idx = np.random.randint(0, n, size=n)
    parent2_idx = np.random.randint(0, n, size=n)
    
    gamete1 = _produce_gametes(pop.haplotypes[parent1_idx], params.recomb_rate)
    gamete2 = _produce_gametes(pop.haplotypes[parent2_idx], params.recomb_rate)
    
    new_haplotypes = np.stack([gamete1, gamete2], axis=1)
    return Population(new_haplotypes)


def mate_assortative(pop: Population, params: SimParams) -> Population:
    """Positive assortative mating (vectorized)."""
    n = pop.n
    strength = params.assortment_strength
    trait = params.assortment_trait
    
    phenotypes = pop.phenotypes(trait)
    
    parent1_idx = np.random.randint(0, n, size=n)
    parent1_pheno = phenotypes[parent1_idx]
    
    parent2_idx = np.empty(n, dtype=np.int64)
    assort_mask = np.random.random(n) < strength
    
    parent2_idx[~assort_mask] = np.random.randint(0, n, size=(~assort_mask).sum())
    
    if assort_mask.any():
        for pheno in np.unique(parent1_pheno[assort_mask]):
            pheno_mask = phenotypes == pheno
            if not pheno_mask.any():
                continue
            candidates = np.where(pheno_mask)[0]
            need_mate = assort_mask & (parent1_pheno == pheno)
            if need_mate.any():
                parent2_idx[need_mate] = np.random.choice(candidates, size=need_mate.sum())
    
    gamete1 = _produce_gametes(pop.haplotypes[parent1_idx], params.recomb_rate)
    gamete2 = _produce_gametes(pop.haplotypes[parent2_idx], params.recomb_rate)
    
    new_haplotypes = np.stack([gamete1, gamete2], axis=1)
    return Population(new_haplotypes)


def mate_disassortative(pop: Population, params: SimParams) -> Population:
    """Negative assortative mating (vectorized)."""
    n = pop.n
    strength = params.assortment_strength
    trait = params.assortment_trait
    
    phenotypes = pop.phenotypes(trait)
    max_pheno = phenotypes.max()
    
    parent1_idx = np.random.randint(0, n, size=n)
    parent1_pheno = phenotypes[parent1_idx]
    target_pheno = max_pheno - parent1_pheno
    
    parent2_idx = np.empty(n, dtype=np.int64)
    assort_mask = np.random.random(n) < strength
    
    parent2_idx[~assort_mask] = np.random.randint(0, n, size=(~assort_mask).sum())
    
    if assort_mask.any():
        unique_targets = np.unique(target_pheno[assort_mask])
        for target in unique_targets:
            available_phenos = np.unique(phenotypes)
            closest = available_phenos[np.abs(available_phenos - target).argmin()]
            
            candidates = np.where(phenotypes == closest)[0]
            need_mate = assort_mask & (target_pheno == target)
            if need_mate.any() and len(candidates) > 0:
                parent2_idx[need_mate] = np.random.choice(candidates, size=need_mate.sum())
    
    gamete1 = _produce_gametes(pop.haplotypes[parent1_idx], params.recomb_rate)
    gamete2 = _produce_gametes(pop.haplotypes[parent2_idx], params.recomb_rate)
    
    new_haplotypes = np.stack([gamete1, gamete2], axis=1)
    return Population(new_haplotypes)


def get_mating_func(system: MatingSystem):
    return {
        MatingSystem.RANDOM: mate_random,
        MatingSystem.ASSORTATIVE: mate_assortative,
        MatingSystem.DISASSORTATIVE: mate_disassortative,
    }[system]


# =============================================================================
# Simulation Engine
# =============================================================================

@dataclass
class SimResult:
    """Results from a single simulation run."""
    params: SimParams
    generations: List[int] = field(default_factory=list)
    
    p_A: List[float] = field(default_factory=list)
    p_B: List[float] = field(default_factory=list)
    
    G_A: List[Dict[str, float]] = field(default_factory=list)
    G_B: List[Dict[str, float]] = field(default_factory=list)
    
    g: List[Dict[str, float]] = field(default_factory=list)
    
    D: List[float] = field(default_factory=list)
    r_squared: List[float] = field(default_factory=list)
    
    def record(self, gen: int, pop: Population):
        stats = pop.stats()
        self.generations.append(gen)
        self.p_A.append(stats['p_A'])
        self.G_A.append(stats['G_A'])
        
        if pop.n_loci == 2:
            self.p_B.append(stats['p_B'])
            self.G_B.append(stats['G_B'])
            self.g.append(stats['g'])
            self.D.append(stats['D'])
            self.r_squared.append(stats['r_squared'])


def simulate(params: SimParams, seed: Optional[int] = None) -> SimResult:
    """Run a single simulation."""
    if seed is not None:
        np.random.seed(seed)
    
    result = SimResult(params=params)
    pop = Population.from_params(params)
    mate_func = get_mating_func(params.mating_system)
    
    result.record(0, pop)
    
    for gen in range(1, params.n_generations + 1):
        pop = mate_func(pop, params)
        result.record(gen, pop)
    
    return result


def simulate_replicates(params: SimParams) -> List[SimResult]:
    """Run multiple replicate simulations."""
    return [simulate(params, seed=i) for i in range(params.n_replicates)]


# =============================================================================
# Plotting Functions
# =============================================================================

def _plot_with_ci(ax, x, all_y, color, label):
    """Helper: plot individual traces, mean, and 90% CI."""
    all_y = np.array(all_y)
    
    for y in all_y:
        ax.plot(x, y, color=color, alpha=0.15, lw=0.5)
    
    mean_y = np.mean(all_y, axis=0)
    ax.plot(x, mean_y, color=color, lw=2, label=label)
    
    if len(all_y) > 1:
        lo = np.percentile(all_y, 5, axis=0)
        hi = np.percentile(all_y, 95, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2)


def plot_replicates(results: List[SimResult], figsize: Tuple[int, int] = (14, 8)):
    """Plot multiple replicates with mean and confidence interval."""
    if not results:
        return None, None
    
    n_loci = results[0].params.n_loci
    gens = results[0].generations
    
    if n_loci == 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        _plot_with_ci(axes[0], gens, [r.p_A for r in results], COLOR_A, 'p(A)')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('p(A)')
        axes[0].set_title(f'Allele Frequency ({len(results)} replicates)')
        axes[0].set_ylim(0, 1)
        
        het_freqs = [[g['Aa'] for g in r.G_A] for r in results]
        _plot_with_ci(axes[1], gens, het_freqs, COLORS_A['Aa'], 'G(Aa)')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('G(Aa)')
        axes[1].set_title('Heterozygote Frequency')
        axes[1].set_ylim(0, 1)
        
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        _plot_with_ci(axes[0, 0], gens, [r.p_A for r in results], COLOR_A, 'p(A)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('p(A)')
        
        _plot_with_ci(axes[0, 1], gens, [r.p_B for r in results], COLOR_B, 'p(B)')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('p(B)')
        
        het_A = [[g['Aa'] for g in r.G_A] for r in results]
        _plot_with_ci(axes[0, 2], gens, het_A, COLORS_A['Aa'], 'G(Aa)')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].set_xlabel('Generation')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('G(Aa)')
        
        g_AB = [[gg['AB'] for gg in r.g] for r in results]
        _plot_with_ci(axes[1, 0], gens, g_AB, COLORS_G['AB'], 'g(AB)')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('g(AB)')
        
        _plot_with_ci(axes[1, 1], gens, [r.D for r in results], '#2ca02c', 'D')
        axes[1, 1].axhline(0, color='gray', ls='--', lw=0.5)
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('D')
        axes[1, 1].set_title('Linkage Disequilibrium')
        
        _plot_with_ci(axes[1, 2], gens, [r.r_squared for r in results], '#9467bd', 'r²')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_xlabel('Generation')
        axes[1, 2].set_ylabel('r²')
        axes[1, 2].set_title('LD (r²)')
    
    p = results[0].params
    title = f"N={p.pop_size}, {p.n_loci} locus, {p.mating_system.value} mating, {len(results)} replicates"
    fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    return fig, axes


def compare_scenarios(scenarios: List[Tuple[str, SimParams]], 
                      metric: str = 'p_A',
                      figsize: Tuple[int, int] = (10, 6)):
    """Compare multiple parameter scenarios on one plot."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    
    for (name, params), color in zip(scenarios, colors):
        results = simulate_replicates(params)
        gens = results[0].generations
        
        if metric in ['p_A', 'p_B', 'D', 'r_squared']:
            all_vals = [getattr(r, metric) for r in results]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if params.n_replicates > 1:
            for vals in all_vals:
                ax.plot(gens, vals, color=color, alpha=0.15, lw=0.5)
        
        mean_vals = np.mean(all_vals, axis=0)
        ax.plot(gens, mean_vals, color=color, lw=2, label=name)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel(metric)
    ax.legend()
    
    if metric == 'D':
        ax.axhline(0, color='gray', ls='--', lw=0.5)
    
    plt.tight_layout()
    return fig, ax


# =============================================================================
# Interactive Player
# =============================================================================

def make_player(result: SimResult):
    """Create interactive player for simulation results."""
    import ipywidgets as widgets
    from IPython.display import clear_output
    
    n_gen = len(result.generations) - 1
    n_loci = result.params.n_loci
    
    play = widgets.Play(
        value=0, min=0, max=n_gen,
        step=1, interval=100,
        description="Play"
    )
    slider = widgets.IntSlider(
        value=0, min=0, max=n_gen,
        description='Generation',
        continuous_update=True,
        layout=widgets.Layout(width='600px')
    )
    speed = widgets.IntSlider(
        value=100, min=20, max=500, step=20,
        description='Speed (ms)',
        layout=widgets.Layout(width='200px')
    )
    
    widgets.jslink((play, 'value'), (slider, 'value'))
    
    def update_speed(change):
        play.interval = change['new']
    speed.observe(update_speed, names='value')
    
    out = widgets.Output()
    
    def draw(gen):
        with out:
            clear_output(wait=True)
            
            if n_loci == 1:
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                
                # Allele frequency trajectory
                axes[0].plot(result.generations[:gen+1], result.p_A[:gen+1], 
                            color=COLOR_A, lw=2)
                axes[0].axhline(result.p_A[0], color='gray', ls='--', lw=0.5, alpha=0.5)
                axes[0].set_xlim(0, n_gen)
                axes[0].set_ylim(0, 1)
                axes[0].set_xlabel('Generation')
                axes[0].set_ylabel('p(A)')
                axes[0].set_title('Allele Frequency')
                axes[0].axvline(gen, color='red', ls='-', lw=1, alpha=0.5)
                
                # Genotype frequencies (bar)
                G = result.G_A[gen]
                genos = ['aa', 'Aa', 'AA']
                colors = [COLORS_A[g] for g in genos]
                axes[1].bar(genos, [G[g] for g in genos], color=colors, edgecolor='black', lw=0.5)
                axes[1].set_ylim(0, 1)
                axes[1].set_ylabel('Frequency')
                axes[1].set_title(f'Genotype Frequencies (Gen {gen})')
                
                # HW expectation
                p = result.p_A[gen]
                hw = [(1-p)**2, 2*p*(1-p), p**2]
                for i, exp in enumerate(hw):
                    axes[1].plot(i, exp, 'k_', markersize=15, mew=2)
                axes[1].annotate('— HW expected', (1.5, 0.92), fontsize=9)
                
                # Genotype trajectory
                for geno in genos:
                    freqs = [g[geno] for g in result.G_A[:gen+1]]
                    axes[2].plot(result.generations[:gen+1], freqs, 
                                color=COLORS_A[geno], lw=2, label=geno)
                axes[2].set_xlim(0, n_gen)
                axes[2].set_ylim(0, 1)
                axes[2].set_xlabel('Generation')
                axes[2].set_ylabel('Frequency')
                axes[2].set_title('Genotype Trajectories')
                axes[2].legend(loc='upper right')
                axes[2].axvline(gen, color='red', ls='-', lw=1, alpha=0.5)
                
            else:  # 2 loci
                fig, axes = plt.subplots(2, 3, figsize=(14, 8))
                
                # Allele frequencies
                axes[0,0].plot(result.generations[:gen+1], result.p_A[:gen+1], 
                              color=COLOR_A, lw=2, label='p(A)')
                axes[0,0].plot(result.generations[:gen+1], result.p_B[:gen+1], 
                              color=COLOR_B, lw=2, label='p(B)')
                axes[0,0].axhline(result.p_A[0], color=COLOR_A, ls='--', lw=0.5, alpha=0.3)
                axes[0,0].axhline(result.p_B[0], color=COLOR_B, ls='--', lw=0.5, alpha=0.3)
                axes[0,0].set_xlim(0, n_gen)
                axes[0,0].set_ylim(0, 1)
                axes[0,0].set_xlabel('Generation')
                axes[0,0].set_ylabel('Frequency')
                axes[0,0].set_title('Allele Frequencies')
                axes[0,0].legend()
                axes[0,0].axvline(gen, color='gray', ls='-', lw=1, alpha=0.5)
                
                # Genotypes A (bar) - Blues
                G_A = result.G_A[gen]
                genos_A = ['aa', 'Aa', 'AA']
                colors_A = [COLORS_A[g] for g in genos_A]
                axes[0,1].bar(genos_A, [G_A[g] for g in genos_A], 
                             color=colors_A, edgecolor='black', lw=0.5)
                axes[0,1].set_ylim(0, 1)
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].set_title(f'Genotypes A (Gen {gen})')
                
                # HW expectation A
                p = result.p_A[gen]
                hw_A = [(1-p)**2, 2*p*(1-p), p**2]
                for i, exp in enumerate(hw_A):
                    axes[0,1].plot(i, exp, 'k_', markersize=12, mew=2)
                
                # Genotypes B (bar) - Oranges
                G_B = result.G_B[gen]
                genos_B = ['bb', 'Bb', 'BB']
                colors_B = [COLORS_B[g] for g in genos_B]
                axes[0,2].bar(genos_B, [G_B[g] for g in genos_B], 
                             color=colors_B, edgecolor='black', lw=0.5)
                axes[0,2].set_ylim(0, 1)
                axes[0,2].set_ylabel('Frequency')
                axes[0,2].set_title(f'Genotypes B (Gen {gen})')
                
                # HW expectation B
                q = result.p_B[gen]
                hw_B = [(1-q)**2, 2*q*(1-q), q**2]
                for i, exp in enumerate(hw_B):
                    axes[0,2].plot(i, exp, 'k_', markersize=12, mew=2)
                
                # Gamete frequencies (bar)
                g_curr = result.g[gen]
                gametes = ['AB', 'Ab', 'aB', 'ab']
                colors_gam = [COLORS_G[gam] for gam in gametes]
                axes[1,0].bar(gametes, [g_curr[gam] for gam in gametes], 
                             color=colors_gam, edgecolor='black', lw=0.5)
                axes[1,0].set_ylim(0, 1)
                axes[1,0].set_ylabel('Frequency')
                axes[1,0].set_title(f'Gamete Frequencies (Gen {gen})')
                
                # LE expectation
                p_A, p_B = result.p_A[gen], result.p_B[gen]
                le_exp = [p_A*p_B, p_A*(1-p_B), (1-p_A)*p_B, (1-p_A)*(1-p_B)]
                for i, exp in enumerate(le_exp):
                    axes[1,0].plot(i, exp, 'k_', markersize=12, mew=2)
                axes[1,0].annotate('— LE expected', (2.2, 0.92), fontsize=9)
                
                # D trajectory
                axes[1,1].plot(result.generations[:gen+1], result.D[:gen+1], 
                              color='#2ca02c', lw=2)
                axes[1,1].axhline(0, color='gray', ls='--', lw=0.5)
                axes[1,1].axhline(result.D[0], color='#2ca02c', ls='--', lw=0.5, alpha=0.3)
                axes[1,1].set_xlim(0, n_gen)
                D_bound = max(abs(min(result.D)), abs(max(result.D)), 0.05)
                axes[1,1].set_ylim(-D_bound*1.1, D_bound*1.1)
                axes[1,1].set_xlabel('Generation')
                axes[1,1].set_ylabel('D')
                axes[1,1].set_title(f'Linkage Disequilibrium (D = {result.D[gen]:.4f})')
                axes[1,1].axvline(gen, color='gray', ls='-', lw=1, alpha=0.5)
                
                # r² trajectory
                axes[1,2].plot(result.generations[:gen+1], result.r_squared[:gen+1], 
                              color='#9467bd', lw=2)
                axes[1,2].set_xlim(0, n_gen)
                axes[1,2].set_ylim(0, 1)
                axes[1,2].set_xlabel('Generation')
                axes[1,2].set_ylabel('r²')
                axes[1,2].set_title(f'LD r² = {result.r_squared[gen]:.4f}')
                axes[1,2].axvline(gen, color='gray', ls='-', lw=1, alpha=0.5)
            
            pr = result.params
            title = f"N={pr.pop_size}, {pr.mating_system.value} mating"
            if n_loci == 2:
                title += f", r={pr.recomb_rate}"
            fig.suptitle(title, fontsize=12, y=1.02)
            
            plt.tight_layout()
            plt.show()
    
    def on_change(change):
        draw(change['new'])
    slider.observe(on_change, names='value')
    
    draw(0)
    
    controls = widgets.HBox([play, slider, speed])
    return widgets.VBox([controls, out])


# =============================================================================
# Multi-Allele Selection Model (Deterministic)
# =============================================================================
#
# Infinite-population model for k alleles at one locus under viability
# selection with random mating (HW genotype frequencies each generation
# before selection acts).
#
# Students specify:
#   - initial allele frequencies (k alleles, must sum to 1)
#   - a symmetric fitness matrix W[i,j] for diploid genotypes
#
# Each generation:
#   1. Form HW genotype frequencies: G_ij = p_i * p_j
#   2. Apply viability selection:    G_ij' = W_ij * G_ij / w_bar
#   3. Derive new allele freqs:      p_i'  = sum_j G_ij'
#      (summing over one index of the diploid freq table gives
#       the frequency of allele i among gametes)
#
# This cleanly separates drift (the existing individual-based code)
# from deterministic selection dynamics.
#
# Nomenclature for 3 alleles:
#   Alleles: A1, A2, A3
#   Genotypes: A1A1, A1A2, A1A3, A2A2, A2A3, A3A3
# =============================================================================


# Default color palette for up to 6 alleles
COLORS_ALLELES = [
    '#1f78b4',  # blue
    '#e31a1c',  # red
    '#33a02c',  # green
    '#ff7f00',  # orange
    '#6a3d9a',  # purple
    '#b15928',  # brown
]

# Genotype colors (for 3 alleles = 6 genotypes)
COLORS_GENOTYPES = [
    '#08306b',  # A1A1 — dark blue
    '#6baed6',  # A1A2 — light blue
    '#74c476',  # A1A3 — green-blue
    '#c51b8a',  # A2A2 — magenta
    '#fa9fb5',  # A2A3 — pink
    '#238b45',  # A3A3 — dark green
]


@dataclass
class SelectionParams:
    """
    Parameters for multi-allele deterministic selection model.

    Attributes
    ----------
    n_alleles : int
        Number of alleles (default 3).
    freqs : list of float
        Initial allele frequencies (must sum to 1).
    fitness : dict mapping genotype string -> float, OR numpy array
        Diploid genotype fitnesses. Can be specified as:
          - dict with keys like 'A1A1', 'A1A2', etc.
          - k×k symmetric numpy array W[i,j]
        If None, all fitnesses default to 1.0 (neutral).
    allele_labels : list of str or None
        Labels for alleles (default: ['A1', 'A2', ...]).
    n_generations : int
        Number of generations to simulate.
    """
    n_alleles: int = 3
    freqs: Optional[List[float]] = None
    fitness: Optional[object] = None  # dict or ndarray
    allele_labels: Optional[List[str]] = None
    n_generations: int = 200

    def __post_init__(self):
        k = self.n_alleles

        # --- allele labels ---
        if self.allele_labels is None:
            self.allele_labels = [f'A{i+1}' for i in range(k)]
        if len(self.allele_labels) != k:
            raise ValueError(
                f"allele_labels length ({len(self.allele_labels)}) "
                f"must match n_alleles ({k})"
            )

        # --- initial frequencies ---
        if self.freqs is None:
            self.freqs = [1.0 / k] * k
        self.freqs = list(self.freqs)
        if len(self.freqs) != k:
            raise ValueError(
                f"freqs length ({len(self.freqs)}) must match n_alleles ({k})"
            )
        total = sum(self.freqs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"freqs must sum to 1 (got {total})")
        # Normalize to machine precision
        self.freqs = [f / total for f in self.freqs]

        # --- fitness matrix ---
        self._W = self._build_fitness_matrix()

    def _build_fitness_matrix(self) -> np.ndarray:
        k = self.n_alleles
        labels = self.allele_labels

        if self.fitness is None:
            return np.ones((k, k))

        if isinstance(self.fitness, np.ndarray):
            W = np.array(self.fitness, dtype=float)
            if W.shape != (k, k):
                raise ValueError(f"fitness array must be {k}x{k}")
            # Enforce symmetry
            if not np.allclose(W, W.T):
                raise ValueError("fitness matrix must be symmetric")
            return W

        if isinstance(self.fitness, dict):
            W = np.ones((k, k))
            for key, val in self.fitness.items():
                i, j = self._parse_genotype_key(key, labels)
                W[i, j] = val
                W[j, i] = val
            return W

        raise TypeError("fitness must be dict, numpy array, or None")

    @staticmethod
    def _parse_genotype_key(key: str, labels: List[str]) -> Tuple[int, int]:
        """Parse a genotype string like 'A1A2' into index pair (0, 1)."""
        # Sort labels longest-first so 'A10' is matched before 'A1'
        sorted_labels = sorted(enumerate(labels), key=lambda x: -len(x[1]))
        remaining = key
        found = []
        for idx, lab in sorted_labels:
            while remaining.startswith(lab):
                found.append(idx)
                remaining = remaining[len(lab):]
                if len(found) == 2:
                    break
            if len(found) == 2:
                break
        if len(found) != 2 or remaining:
            raise ValueError(
                f"Cannot parse genotype '{key}' with labels {labels}"
            )
        return (found[0], found[1])

    def genotype_labels(self) -> List[Tuple[int, int, str]]:
        """Return list of (i, j, label_string) for unique genotypes (i <= j)."""
        k = self.n_alleles
        labs = self.allele_labels
        result = []
        for i in range(k):
            for j in range(i, k):
                result.append((i, j, f'{labs[i]}{labs[j]}'))
        return result

    @property
    def W(self) -> np.ndarray:
        return self._W


@dataclass
class SelectionResult:
    """
    Results from a deterministic multi-allele selection simulation.

    Attributes
    ----------
    params : SelectionParams
    generations : list of int
    allele_freqs : list of ndarray, each shape (k,)
        Allele frequencies per generation.
    genotype_freqs : list of ndarray, each shape (k, k)
        Diploid genotype frequencies per generation (before selection).
    w_bar : list of float
        Mean population fitness per generation.
    """
    params: SelectionParams
    generations: List[int] = field(default_factory=list)
    allele_freqs: List[np.ndarray] = field(default_factory=list)
    genotype_freqs: List[np.ndarray] = field(default_factory=list)
    w_bar: List[float] = field(default_factory=list)


def simulate_selection(params: SelectionParams) -> SelectionResult:
    """
    Run deterministic multi-allele viability selection simulation.

    Each generation:
      1. HW genotype frequencies from current allele freqs
      2. Viability selection (multiply by W, normalize)
      3. New allele freqs by marginalizing genotype freqs
    """
    k = params.n_alleles
    W = params.W
    p = np.array(params.freqs)

    result = SelectionResult(params=params)

    for gen in range(params.n_generations + 1):
        # HW genotype frequencies (random mating, infinite pop)
        G = np.outer(p, p)

        # Mean fitness
        w_bar = np.sum(W * G)

        # Record state (before selection for gen > 0, or initial for gen 0)
        result.generations.append(gen)
        result.allele_freqs.append(p.copy())
        result.genotype_freqs.append(G.copy())
        result.w_bar.append(w_bar)

        # Selection + new allele freqs
        G_prime = (W * G) / w_bar
        # Allele freq = marginal sum: p_i' = sum_j G'_ij
        # But since G' is symmetric for diploids, summing either axis works;
        # however we stored G_ij = p_i * p_j so G'_ij = W_ij * p_i * p_j / w_bar.
        # The new frequency of allele i is sum over all genotypes carrying i:
        #   p_i' = G'_ii + 0.5 * sum_{j != i} (G'_ij + G'_ji)
        # Since G' is symmetric: p_i' = G'_ii + sum_{j != i} G'_ij = sum_j G'_ij
        p = G_prime.sum(axis=1)

        # Safety: normalize and clip
        p = np.clip(p, 0, 1)
        p /= p.sum()

    return result


# =============================================================================
# Selection Plotting
# =============================================================================

def plot_selection(result: SelectionResult, figsize: Tuple[int, int] = (14, 5)):
    """
    Plot allele frequency trajectories, genotype frequencies, and mean fitness.
    """
    params = result.params
    k = params.n_alleles
    labels = params.allele_labels
    gens = result.generations

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Panel 1: Allele frequency trajectories ---
    ax = axes[0]
    for i in range(k):
        freqs_i = [af[i] for af in result.allele_freqs]
        color = COLORS_ALLELES[i % len(COLORS_ALLELES)]
        ax.plot(gens, freqs_i, color=color, lw=2, label=labels[i])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Allele frequency')
    ax.set_title('Allele Frequencies')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='best')

    # --- Panel 2: Genotype frequencies (stacked area) ---
    ax = axes[1]
    geno_info = params.genotype_labels()
    geno_trajs = []
    geno_labels = []
    geno_colors = []
    for idx, (i, j, lab) in enumerate(geno_info):
        traj = []
        for G in result.genotype_freqs:
            if i == j:
                traj.append(G[i, j])
            else:
                traj.append(G[i, j] + G[j, i])  # both orderings
        geno_trajs.append(traj)
        geno_labels.append(lab)
        geno_colors.append(COLORS_GENOTYPES[idx % len(COLORS_GENOTYPES)])

    ax.stackplot(gens, *geno_trajs, labels=geno_labels, colors=geno_colors,
                 alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Genotype frequency')
    ax.set_title('Genotype Frequencies')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=8)

    # --- Panel 3: Mean fitness ---
    ax = axes[2]
    ax.plot(gens, result.w_bar, color='#333333', lw=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean fitness (w̄)')
    ax.set_title('Mean Population Fitness')
    if result.w_bar:
        ymin = min(result.w_bar)
        ymax = max(result.w_bar)
        margin = max((ymax - ymin) * 0.1, 0.01)
        ax.set_ylim(ymin - margin, ymax + margin)

    fig.suptitle(
        f'{k}-allele selection model, {params.n_generations} generations',
        fontsize=12
    )
    plt.tight_layout()
    return fig, axes


def plot_selection_trajectory(result: SelectionResult,
                              figsize: Tuple[int, int] = (7, 6)):
    """
    Ternary-style allele frequency trajectory for 3 alleles.
    Uses a 2D simplex projection (equilateral triangle).
    Only works when n_alleles == 3.
    """
    if result.params.n_alleles != 3:
        raise ValueError("Ternary plot requires exactly 3 alleles")

    labels = result.params.allele_labels

    # Simplex corners (equilateral triangle)
    # A1 = top, A2 = bottom-left, A3 = bottom-right
    corners = np.array([
        [0.5, np.sqrt(3)/2],   # A1 (top)
        [0.0, 0.0],            # A2 (bottom-left)
        [1.0, 0.0],            # A3 (bottom-right)
    ])

    def to_xy(freqs):
        """Convert (p1, p2, p3) to (x, y) via barycentric coords."""
        f = np.array(freqs)
        return f @ corners

    fig, ax = plt.subplots(figsize=figsize)

    # Draw triangle
    triangle = plt.Polygon(corners, fill=False, edgecolor='#888888',
                           lw=1.5, zorder=1)
    ax.add_patch(triangle)

    # Grid lines (10% increments)
    for frac in np.arange(0.1, 1.0, 0.1):
        for pair in [(0, 1), (0, 2), (1, 2)]:
            i, j = pair
            k_idx = 3 - i - j  # the third index
            # Line of constant p_i = frac
            f_start = np.zeros(3)
            f_start[i] = frac
            f_start[j] = 1 - frac
            f_start[k_idx] = 0
            f_end = np.zeros(3)
            f_end[i] = frac
            f_end[j] = 0
            f_end[k_idx] = 1 - frac
            xy_s = to_xy(f_start)
            xy_e = to_xy(f_end)
            ax.plot([xy_s[0], xy_e[0]], [xy_s[1], xy_e[1]],
                    color='#dddddd', lw=0.5, zorder=0)

    # Trajectory
    xy_traj = np.array([to_xy(af) for af in result.allele_freqs])
    ax.plot(xy_traj[:, 0], xy_traj[:, 1], color='#333333', lw=1.5,
            alpha=0.7, zorder=2)
    ax.plot(xy_traj[0, 0], xy_traj[0, 1], 'o', color='#2ca02c',
            ms=8, zorder=3, label='Start')
    ax.plot(xy_traj[-1, 0], xy_traj[-1, 1], 's', color='#e31a1c',
            ms=8, zorder=3, label='End')

    # Color trajectory by generation
    n = len(xy_traj)
    for i_seg in range(n - 1):
        frac = i_seg / max(n - 1, 1)
        color = plt.cm.viridis(frac)
        ax.plot(xy_traj[i_seg:i_seg+2, 0], xy_traj[i_seg:i_seg+2, 1],
                color=color, lw=2, zorder=2)

    # Corner labels
    offset = 0.04
    ax.text(corners[0][0], corners[0][1] + offset, labels[0],
            ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.text(corners[1][0] - offset, corners[1][1] - offset, labels[1],
            ha='right', va='top', fontsize=13, fontweight='bold')
    ax.text(corners[2][0] + offset, corners[2][1] - offset, labels[2],
            ha='left', va='top', fontsize=13, fontweight='bold')

    ax.legend(loc='upper right')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Allele Frequency Trajectory (Simplex)', fontsize=12)

    plt.tight_layout()
    return fig, ax


def make_selection_player(result: SelectionResult):
    """Create interactive player for selection simulation results."""
    import ipywidgets as widgets
    from IPython.display import clear_output

    params = result.params
    k = params.n_alleles
    labels = params.allele_labels
    n_gen = len(result.generations) - 1

    play = widgets.Play(
        value=0, min=0, max=n_gen,
        step=1, interval=100, description="Play"
    )
    slider = widgets.IntSlider(
        value=0, min=0, max=n_gen,
        description='Generation',
        continuous_update=True,
        layout=widgets.Layout(width='600px')
    )
    speed = widgets.IntSlider(
        value=100, min=20, max=500, step=20,
        description='Speed (ms)',
        layout=widgets.Layout(width='200px')
    )

    widgets.jslink((play, 'value'), (slider, 'value'))

    def update_speed(change):
        play.interval = change['new']
    speed.observe(update_speed, names='value')

    out = widgets.Output()

    geno_info = params.genotype_labels()

    def draw(gen):
        with out:
            clear_output(wait=True)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

            # --- Allele frequency trajectories ---
            ax = axes[0]
            for i in range(k):
                freqs_i = [af[i] for af in result.allele_freqs[:gen+1]]
                color = COLORS_ALLELES[i % len(COLORS_ALLELES)]
                ax.plot(result.generations[:gen+1], freqs_i,
                        color=color, lw=2, label=labels[i])
            ax.axvline(gen, color='red', ls='-', lw=1, alpha=0.5)
            ax.set_xlim(0, n_gen)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Frequency')
            ax.set_title('Allele Frequencies')
            ax.legend(loc='best', fontsize=9)

            # --- Genotype bar chart ---
            ax = axes[1]
            G = result.genotype_freqs[gen]
            geno_vals = []
            geno_labs = []
            geno_cols = []
            for idx, (i, j, lab) in enumerate(geno_info):
                val = G[i, j] + (G[j, i] if i != j else 0)
                geno_vals.append(val)
                geno_labs.append(lab)
                geno_cols.append(
                    COLORS_GENOTYPES[idx % len(COLORS_GENOTYPES)]
                )

            bars = ax.bar(geno_labs, geno_vals, color=geno_cols,
                          edgecolor='black', lw=0.5)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Genotype Frequencies (Gen {gen})')

            # HW expected (from current allele freqs)
            p = result.allele_freqs[gen]
            for idx, (i, j, lab) in enumerate(geno_info):
                hw_exp = p[i]**2 if i == j else 2 * p[i] * p[j]
                ax.plot(idx, hw_exp, 'k_', markersize=12, mew=2)
            ax.annotate('— HW expected', (len(geno_info)-1.8, 0.92),
                        fontsize=9)

            # --- Mean fitness trajectory ---
            ax = axes[2]
            ax.plot(result.generations[:gen+1], result.w_bar[:gen+1],
                    color='#333333', lw=2)
            ax.axvline(gen, color='red', ls='-', lw=1, alpha=0.5)
            ax.set_xlim(0, n_gen)
            if result.w_bar:
                ymin = min(result.w_bar)
                ymax = max(result.w_bar)
                margin = max((ymax - ymin) * 0.1, 0.01)
                ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_xlabel('Generation')
            ax.set_ylabel('w̄')
            ax.set_title(f'Mean Fitness (w̄ = {result.w_bar[gen]:.4f})')

            # Fitness table as text below
            W = params.W
            fitness_lines = ['Fitness matrix:']
            header = '       ' + '  '.join(f'{l:>6s}' for l in labels)
            fitness_lines.append(header)
            for i in range(k):
                row = f'{labels[i]:>6s} ' + '  '.join(
                    f'{W[i,j]:6.3f}' for j in range(k)
                )
                fitness_lines.append(row)
            fitness_text = '\n'.join(fitness_lines)
            fig.text(0.72, -0.08, fitness_text, fontsize=9,
                     fontfamily='monospace', va='top')

            plt.tight_layout()
            plt.show()

    def on_change(change):
        draw(change['new'])
    slider.observe(on_change, names='value')

    draw(0)

    controls = widgets.HBox([play, slider, speed])
    return widgets.VBox([controls, out])
