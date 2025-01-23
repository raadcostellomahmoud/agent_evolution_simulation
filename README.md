# 3D Ecosystem Simulation

This repository contains a simulation of a 3D ecosystem where prey, predators, and resources interact. The simulation uses the **Agents.jl** package for agent-based modeling and **CairoMakie.jl** for visualization. Organisms evolve through mutation and reproduction, and their behavior is determined by their genome, energy levels, and surroundings.

## Features
- **Organism Types**: Prey and predators with unique traits (speed, sensing range, energy efficiency).
- **Resources**: Energy sources that prey can consume.
- **Genetic Mutation**: Offspring inherit mutated versions of their parent's genome.
- **Behavior**: Prey avoid predators and move toward resources; predators hunt prey.
- **Visualization**: Dynamic 3D visualization of the ecosystem with energy-based coloring.

## Requirements
To run this simulation, you need the following Julia packages installed:
- `LinearAlgebra`
- `Distributions`
- `CairoMakie`
- `Agents`
- `Colors`
- `ColorSchemes`

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Required Packages**:
   Open a Julia REPL and run:
   ```julia
   using Pkg
   Pkg.add(["LinearAlgebra", "Distributions", "CairoMakie", "Agents", "Colors", "ColorSchemes"])
   ```

3. **Run the Simulation**:
   Load the script in Julia and execute the `make_vid(fig)` function to generate a video of the simulation:
   ```julia
   include("simulation.jl")
   make_vid(fig)
   ```

   The simulation video will be saved as `simulation.mp4`.

## Simulation Details

### Agent Structure
Each organism is an agent defined as follows:
```julia
@agent struct Organism3D(ContinuousAgent{3,Float64})
    species::Symbol  # :prey or :predator
    energy::Float64  # Current energy level
    genome::Vector{Float64}  # [speed, sensing_range, energy_efficiency]
end
```

### Initialization
The ecosystem initializes with:
- A user-defined number of prey, predators, and resources.
- Prey and predators are randomly placed in a 3D space of size `(100, 100, 100)`.

### Behavior
- **Prey**: Move toward nearby resources or away from predators.
- **Predators**: Hunt prey within their sensing range.
- **Reproduction**: Organisms reproduce when their energy exceeds a threshold.
- **Energy Management**: Organisms lose energy based on their speed and efficiency. Prey gain energy by consuming resources, and predators gain energy by consuming prey.

### Visualization
- **Colors**: Organisms are colored based on their energy level (green = high, red = low).
- **Shapes**: Prey are represented as circles, and predators as triangles.
- **Resources**: Displayed as light cyan hexagons.

### Output
The simulation creates an MP4 video (`simulation.mp4`) that visualizes the ecosystem dynamics over time.

## Parameters
You can modify these parameters in the `initialize_model` function:
- `n_prey`: Number of prey.
- `n_predators`: Number of predators.
- `n_resources`: Number of resource pools.
- `extent`: Size of the simulation space.

Example:
```julia
model, resources = initialize_model(n_prey=20, n_predators=5, n_resources=50, extent=(150.0, 150.0, 150.0))
```

## Example Video
The resulting video shows prey avoiding predators, predators hunting prey, and resources being consumed, leading to emergent ecosystem dynamics.

## License
This project is licensed under the MIT License.

## Contributions
Contributions and suggestions are welcome! Please feel free to open issues or submit pull requests.

## Acknowledgments
- [Agents.jl](https://github.com/JuliaDynamics/Agents.jl) for agent-based modeling.
- [CairoMakie.jl](https://github.com/MakieOrg/CairoMakie.jl) for visualization.