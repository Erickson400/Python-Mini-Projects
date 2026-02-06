import random
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Global Constants
ITERATIONS = 10
SWARM_SIZE = 1000
DNA_LENGTH = 20
MAP_SIZE = 20
START_POSITION = 16
GOAL_POSITION = 4

MOVE_TYPE_PROBABILITY = 0.7 # Chances of move type: 0.0 = random exploration moves, 1.0 = target seeking moves
SEEKING_PROBABILITY = 0.3 # Chance for each gene to move toward target in seeking mode

def clamp(n, minn, maxn):
    return max(minn, min(n, maxn))

class Vector:
    def __init__(self, values=None):
        if values is None:
            self.values = [0] * DNA_LENGTH
        else:
            self.values = list(values)

    def randomized(self):
        return Vector([random.randint(-1, 1) for _ in range(DNA_LENGTH)])

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector([a + b for a, b in zip(self.values, other.values)])
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector([a - b for a, b in zip(self.values, other.values)])
        return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector([int(x * scalar) for x in self.values])
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __repr__(self):
        return f"Vector({self.values})"

    def __iter__(self):
        return iter(self.values)

    def clamp(self, minn, maxn):
        return Vector([clamp(v, minn, maxn) for v in self.values])

class Particle:
    def __init__(self):
        self.dna = Vector().randomized()

    def update(self, best_particle):
        r = random.random()
        
        if r < MOVE_TYPE_PROBABILITY:
            # Target Seeking Mode
            new_values = []
            for my_gene, target_gene in zip(self.dna, best_particle.dna):
                if my_gene != target_gene and random.random() < SEEKING_PROBABILITY:
                    my_gene += 1 if my_gene < target_gene else -1
                new_values.append(my_gene)
            self.dna = Vector(new_values)
        else:
            # Random Walk Mode
            new_values = []
            for gene in self.dna:
                gene += random.randint(-1, 1)
                new_values.append(clamp(gene, -1, 1))
            self.dna = Vector(new_values)

    def get_fitness(self):
        current_pos = START_POSITION
        positions = []
        total_distance_penalty = 0
        total_movement_penalty = 0
        
        for gene in self.dna:
            current_pos = clamp(current_pos + gene, 0, MAP_SIZE)
            positions.append(current_pos)
            total_distance_penalty += abs(current_pos - GOAL_POSITION)
            total_movement_penalty += abs(gene)
            
        # Cumulative distance penalty rewards reaching the goal fast.
        # Movement penalty rewards efficiency (using fewer/smaller moves).
        fitness = total_distance_penalty + total_movement_penalty
        return fitness, positions

def plot_move(history):
    """
    Plots the path of 3D coordinates in history using Plotly.
    """
    xs, ys, zs = zip(*history)
    
    fig = go.Figure()
    
    # Path trace
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines+markers',
        marker=dict(size=3, opacity=0.8, color='cyan'),
        line=dict(width=4, color='cyan'),
        name='Path'
    ))
    
    # Start and End points
    fig.add_trace(go.Scatter3d(
        x=[xs[0]], y=[ys[0]], z=[zs[0]],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='End'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=f"3D Point Movement Visualization (Steps: {len(history)-1})",
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()

def plot_fitness(history):
    """
    Plots the fitness progress over iterations using matplotlib.
    """
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o', linestyle='-', color='cyan')
    plt.title("PSO Fitness Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    swarm = [Particle() for _ in range(SWARM_SIZE)]
    
    # Initialize global best tracker
    def find_initial_best(swarm):
        best_p = None
        best_f = float('inf')
        for p in swarm:
            f, path = p.get_fitness()
            if f < best_f:
                best_f = f
                best_p = p
        
        # Store a snapshot of the best state to prevent drift
        f, path = best_p.get_fitness()
        return Vector(best_p.dna.values), f, path

    best_dna, best_fitness, best_path = find_initial_best(swarm)
    fitness_history = [best_fitness]
    
    print(f"Initial Global Best Fitness: {best_fitness}")
    for i in range(ITERATIONS):
        # Create a leader proxy for update calls in this iteration
        # This ensures everyone tries to move toward the same static 'best' point
        leader = type('obj', (object,), {'dna': best_dna})
        
        fitnesses = []
        for particle in swarm:
            # Update particle
            particle.update(leader)
            
            # Update global best snapshots if this particle's current fitness is better
            current_f, current_path = particle.get_fitness()
            fitnesses.append(current_f)
            if current_f < best_fitness:
                best_fitness = current_f
                best_dna = Vector(particle.dna.values)
                best_path = current_path
                
        fitness_history.append(best_fitness)
        print(f"Iteration {i+1}/{ITERATIONS} - Global Best Fitness: {best_fitness}")
        # print(f"Iteration {i+1}/{ITERATIONS} - Global Best Fitness: {best_fitness} | Swarm: {fitnesses}")

    # Final result visualization
    print(f"\nFinal Global Best Fitness: {best_fitness}")
    print(f"Final Path: {best_path}")
    print(f"Final DNA: {best_dna}")
    
    plot_fitness(fitness_history)
