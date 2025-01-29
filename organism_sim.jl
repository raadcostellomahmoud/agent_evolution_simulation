using LinearAlgebra
using Distributions
using CairoMakie
using Agents
using Colors
using ColorSchemes

N_STEPS::Int64 = 10000  # Number of steps to run simulation for

N_PREY::Int64 = 100
N_PREDATORS::Int64 = 3
N_RESOURCES::Int64 = 50
EXTENT::Int64 = 100

μ_RESOURCE_ENERGY::Float64 = 15.0
REPRODUCTION_PROXIMITY::Int64 = 3

INITIAL_PREY_ENERGY::Int64 = 20
INITIAL_PREDATOR_ENERGY::Int64 = 20

NEW_RESOURCE_CHANCE::Float64 = 0.01

RANDOM_WALK_CHANCE::Float64 = 0.2

MUTATION_CHANCE::Float64 = 0.1
MUTATION_STRENGTH::Float64 = 0.1

REPRODUCTION_ENERGY_THRESHOLD::Int64 = 20

# Define the agent type using the correct syntax
@agent struct Organism3D(ContinuousAgent{3,Float64})
    species::Symbol
    energy::Float64
    genome::Vector{Float64} #[speed, sensing_range, energy_efficiency, consumption_range]
end

function initialize_resources()
    # Initialize resource pools
    resources = []
    for _ in 1:N_RESOURCES
        pos = [rand(3) .* EXTENT; rand() * μ_RESOURCE_ENERGY]
        push!(resources, pos)
    end
    return resources
end
resources = initialize_resources()

# Define genetic traits and mutation
function mutate_genome(genome)
    new_genome = copy(genome)
    for i in eachindex(new_genome)
        if rand() < MUTATION_CHANCE
            new_genome[i] += randn() * MUTATION_STRENGTH  # Small random changes
            new_genome[i] = abs(new_genome[i])  # Ensure positive values
        end
    end
    return new_genome
end

# The ability to sense nearby prey (for predators) or resources and predators (for prey)
function sense_nearby!(agent, model)
    nearby = Vector{Union{Organism3D,Nothing}}(nothing, 3)
    count = 1
    for other_agent in allagents(model)
        if count <= length(nearby)
            if agent != other_agent
                if norm(agent.pos .- other_agent.pos) <= agent.genome[2]
                    nearby[count] = other_agent
                    count += 1
                end
            end
        end
    end
    return nearby
end

function prey_behavior!(agent, model)
    # Prey-specific logic (resource finding, avoiding predators)
    speed = agent.genome[1]
    direction = zeros(3)
    for predator in sense_nearby!(agent, model)
        if typeof(predator) == Organism3D
            if predator.species == :predator
                direction = normalize(agent.pos .- predator.pos)
            end
        end
    end
    # Resource-driven movement
    if direction == zeros(3) && !isempty(resources)
        closest_resource = argmin(norm(agent.pos .- r[1:3]) for r in resources)
        resource_pos = resources[closest_resource][1:3]
        norm_resource_pos = norm(agent.pos - resource_pos)
        if norm_resource_pos ≤ agent.genome[2]
            direction = normalize(resource_pos .- agent.pos)
            if norm_resource_pos <= speed
                speed = norm_resource_pos
            end
        end
    end
    return direction, speed
end

function predator_behavior!(agent, model)
    # Predator-specific logic (hunting prey)
    direction = zeros(3)
    speed = agent.genome[1]
    for prey in sense_nearby!(agent, model)
        if typeof(prey) == Organism3D && prey.species == :prey
            direction += normalize(prey.pos .- agent.pos)
            if norm(direction) < speed
                speed = norm(direction)
            end
        end
    end
    return direction, speed
end


function reproduce!(parent1, parent2, model)
    # Ensure both parents have enough energy to reproduce
    if parent1.energy < REPRODUCTION_ENERGY_THRESHOLD || parent2.energy < REPRODUCTION_ENERGY_THRESHOLD
        return
    end

    # Split energy between parents
    parent1.energy /= 2
    parent2.energy /= 2

    # Create offspring position and velocity
    pos = ntuple(i -> clamp((parent1.pos[i] + parent2.pos[i]) / 2 + randn(), 0, EXTENT), 3)
    vel = Tuple(randn(3))

    # Mix genomes of both parents and apply mutation
    mixed_genome = (parent1.genome .+ parent2.genome) ./ 2
    new_genome = mutate_genome(mixed_genome)

    add_agent!(
        pos,
        model,
        vel,
        parent1.species,  # Assuming offspring inherits species from parent1
        parent1.energy,   # Offspring inherits energy from parent1
        new_genome
    )
end

function hunt!(predator, model)
    nearby = sense_nearby!(predator, model)
    for prey in nearby
        if typeof(prey) == Organism3D && prey.species == :prey
            if norm(predator.pos - prey.pos) <= predator.genome[4]  # Within consumption range
                predator.energy += prey.energy  # Gain energy from prey
                remove_agent!(prey, model)
                break
            end
        end
    end
end

function agent_step!(agent, model)
    direction, speed = agent.species == :prey ? prey_behavior!(agent, model) : predator_behavior!(agent, model)
    if norm(direction) > 0
        agent.vel = normalize(direction)
    elseif rand() < RANDOM_WALK_CHANCE
        agent.vel = normalize(randn(3))
    end

    proposed_pos = ntuple(i -> clamp(agent.pos[i] .+ agent.vel[i] * speed, 0, EXTENT), 3)

    # Boundary behavior: reflect direction if at boundary
    newvel = Vector(agent.vel)
    for i in 1:3
        if proposed_pos[i] <=0 || proposed_pos[i] >= EXTENT
            newvel[i] *= -1  # Reverse velocity component
        end
    end
    agent.vel = SVector(newvel...)

    new_pos = ntuple(i -> clamp(agent.pos[i] .+ agent.vel[i] * speed, 0., EXTENT), 3)
    move_agent!(agent, new_pos, model)

    agent.energy -= speed^2 / agent.genome[3]
    
    # Handle reproduction with nearby agents
    near_agents = nearby_agents(agent, model, REPRODUCTION_PROXIMITY)  # Function to find nearby agents
    for other_agent in near_agents
        if other_agent.species == agent.species && other_agent != agent
            reproduce!(agent, other_agent, model)
            break  # Only reproduce with one nearby agent per step
        end
    end

    # Handle predation
    if agent.species == :predator
        hunt!(agent, model)
    end

end

function model_step!(model)
    # Check if any resource is consumed by nearby prey
    for agent in allagents(model)
        if agent.species == :prey
            for (i, resource) in enumerate(resources)
                if norm(agent.pos - SVector(resource[1:3]...)) <= agent.genome[4]  # Within consumption range
                    agent.energy += max(0, min(resource[4], 20))  # Gain energy by consuming resource
                    resource[4] = resource[4] - 20  # Reduce resource level
                    if resource[4] <= 0
                        deleteat!(resources, i)  # Remove the resource
                        break
                    end
                end
            end
        end
    end

    # Respawn new resources at random locations
    if rand() <= NEW_RESOURCE_CHANCE  # 10% chance per step
        pos = [rand(3) .* spacesize(model)..., rand() * μ_RESOURCE_ENERGY]
        push!(resources, pos)
    end

    # Remove dead agents
    for agent in allagents(model)
        if agent.energy <= 0
            remove_agent!(agent, model)
        end
    end
end

# Initialize model with both species
function initialize_model()
    space = ContinuousSpace((EXTENT, EXTENT, EXTENT))
    properties = Dict()
    model = AgentBasedModel(Organism3D, space; properties=properties)

    # Initialize prey
    for _ in 1:N_PREY
        pos = Tuple(EXTENT / 2 .+ randn(3) .* EXTENT / 10)
        vel = Tuple(randn(3))

        # Genome: [speed, sensing_range, energy_efficiency, consumption_range]
        genome = abs.([randn() * 0.1, randn() * 30, randn() * 300, 4])
        add_agent!(
            pos,
            model,
            vel,
            :prey,
            INITIAL_PREY_ENERGY,  # initial prey energy
            genome
        )
    end

    # Initialize predators
    for _ in 1:N_PREDATORS
        pos = Tuple(EXTENT / 2 .+ randn(3) .* EXTENT / 10)
        vel::Tuple{Float32,Float32,Float32} = 0.001 .* Tuple(randn(3))

        # Genome: [speed, sensing_range, energy_efficiency, consumption_range]
        genome = abs.([randn() * 0.2, randn() * 30, randn() * 300, 4])
        add_agent!(
            pos,
            model,
            vel,
            :predator,
            INITIAL_PREDATOR_ENERGY,  # initial energy
            genome
        )
    end

    return model
end

model= initialize_model()



# Improved color function with gradient based on energy
function get_agent_color(agent)
    # Green to red gradient based on energy
    base_color = RGB(1.0 - min(agent.energy / 50.0, 1.0), min(agent.energy / 50.0, 1.0), 0.0)
    return RGBAf(base_color.r, base_color.g, base_color.b, 0.7)
end

# Triangles for predators, circles for prey
function get_agent_shape(agent)
    if agent.species == :prey
        return :circle
    elseif agent.species == :predator
        return :dtriangle
    else
        return B
    end
end

# Improved color for resources
function get_resource_color()
    return RGBAf(0.8, 0.8, 1.0, 0.5)  # Soft light cyan for resources
end

# Draw resources with improved color
function draw_resources!(resources::Vector{Any})
    for resource in resources
        scatter!(ax, [resource[1]], [resource[2]], [resource[3]], color=get_resource_color(), markersize=10, marker=:hexagon, strokecolor=:black, strokewidth=1)
    end
end

# Update plot for each step
function update_plot!(ax, model, resources::Vector{Any}, step::Int64)
    empty!(ax)  # Clear the axis for the next frame
    positions = [agent.pos for agent in allagents(model)]
    if !isempty(positions)
        x = [p[1] for p in positions]
        y = [p[2] for p in positions]
        z = [p[3] for p in positions]
        colors = [get_agent_color(agent) for agent in allagents(model)]
        markers = [get_agent_shape(agent) for agent in allagents(model)]
        scatter!(ax, x, y, z, marker=markers, color=colors, markersize=8, strokecolor=:black, strokewidth=1)
    end
    draw_resources!(resources)
    ax.title = "Step: $step"
end

# Track population counts
predator_counts = Int[]  # To store predator counts at each step
prey_counts = Int[]      # To store prey counts at each step

# Function to count predators and prey
function count_agents(model)
    predators = count(agent -> agent.species == :predator, allagents(model))
    prey = count(agent -> agent.species == :prey, allagents(model))
    return predators, prey
end

# Update the 2D population plot
function update_population_plot!(pop_ax, step::Int64)
    empty!(pop_ax)  # Clear the axis for the next frame
    lines!(pop_ax, 1:step, predator_counts, color=:red, label="Predators")
    lines!(pop_ax, 1:step, prey_counts, color=:green, label="Prey")
    xlims!(pop_ax, 0, step)
    ylims!(pop_ax, 0, max(predator_counts..., prey_counts...) + 5)
end

# Set up visualization 
fig = Figure(size=(800, 800))
# Create a grid layout with two rows
ga = fig[2:12, 1:12] = GridLayout()
gb = fig[1:3, 12:18] = GridLayout()
gc = fig[4:5, 15:16] = GridLayout()  # Legend area

ax = Axis3(ga[1,1], aspect=:data, title="3D Ecosystem Simulation",
    xlabel="X", ylabel="Y", zlabel="Z",
    # titlecolor=:white,
    # xgridcolor=:white,
    # ygridcolor=:white,
    # zgridcolor=:white,
    # backgroundcolor=:black,
    # xspinecolor_1=:white,
    # yspinecolor_1=:white,
    # zspinecolor_1=:white,
    # xspinecolor_2=:white,
    # yspinecolor_2=:white,
    # zspinecolor_2=:white,
    # xspinecolor_3=:white,
    # yspinecolor_3=:white,
    # zspinecolor_3=:white,
)
pop_ax = Axis(gb[1,1], title="Population Over Time",
              xlabel="Steps", ylabel="Population")

# Create separate legend
legend = Legend(gc[1,1],
               [LineElement(color=:red), LineElement(color=:green)],
               ["Predators", "Prey"],
               "Population Types",
               backgroundcolor=:white,
               padding=(10, 10, 10, 10))

# Now, both axes should have equal dimensions
function make_vid(fig)
    xlims!(ax, 0, EXTENT)
    ylims!(ax, 0, EXTENT)
    zlims!(ax, 0, EXTENT)

    # Record animation
    record(fig, "simulation_with_sexual_reproduction.mp4", 1:N_STEPS, framerate=30,) do frame_number
        step!(model, agent_step!, model_step!)  # Simulate one step of the model

        # Update predator-prey counts
        predators, prey = count_agents(model)
        push!(predator_counts, predators)
        push!(prey_counts, prey)

        update_plot!(ax, model, resources, frame_number)
        update_population_plot!(pop_ax, frame_number)
        print("Step: ", frame_number, "\n")
        print("Predators: ", predators, ", Prey: ", prey, "\n")
        print("Resources: ", length(resources), "\n")
    end
end

make_vid(fig)