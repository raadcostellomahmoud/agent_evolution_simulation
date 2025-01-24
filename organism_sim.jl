using LinearAlgebra
using Distributions
using CairoMakie
using Agents
using Colors
using ColorSchemes

N_STEPS::Int64 = 100  # Number of steps to run simulation for

N_PREY::Int64 = 100
N_PREDATORS::Int64 = 2
N_RESOURCES::Int64 = 30
EXTENT::Int64 = 100

μ_RESOURCE_ENERGY::Float64 = 10.0

INITIAL_PREY_ENERGY::Int64 = 20
INITIAL_PREDATOR_ENERGY::Int64 = 20

NEW_RESOURCE_CHANCE::Float64 = 0.01

RANDOM_WALK_CHANCE::Float64 = 0.2

MUTATION_CHANCE::Float64 = 0.1
MUTATION_STRENGTH::Float64 = 0.1

REPRODUCTION_ENERGY_THRESHOLD::Int64 = 50

# Define the agent type using the correct syntax
@agent struct Organism3D(ContinuousAgent{3,Float64})
    species::Symbol
    energy::Float64
    genome::Vector{Float64} #[speed, sensing_range, energy_efficiency, consumption_range]
end

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

    # Initialize resource pools
    resources = []
    for _ in 1:N_RESOURCES
        pos = [rand(3) .* spacesize(model)..., rand() * μ_RESOURCE_ENERGY]
        push!(resources, pos)
    end

    return model, resources
end

model, resources = initialize_model()

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
    filtered_nearby = filter(!isnothing, nearby[1:max(1, count - 1)])
    return filtered_nearby
end

# Movement and behavior functions
function agent_step!(agent, model)
    speed = agent.genome[1]
    direction = [0.0, 0.0, 0.0]
    if agent.species == :prey
        nearby = sense_nearby!(agent, model)
        for predator in nearby
            if predator.species == :predator
                direction = normalize(agent.pos .- SVector(predator.pos...))
            end
        end
        if direction == [0.0, 0.0, 0.0]
            if length(resources) > 0
                closest_resource = argmin([norm(agent.pos .- r[1:3]) for r in resources])
                resource_pos = resources[closest_resource][1:3]
                nearest_resource_direction = SVector(resource_pos...) .- agent.pos
                norm_nearest_resource_direction = norm(nearest_resource_direction)
                if norm_nearest_resource_direction <= agent.genome[2]
                    direction = normalize(SVector(nearest_resource_direction...))
                    if norm_nearest_resource_direction < speed
                        speed = norm_nearest_resource_direction
                    end
                end
            end
        end

    else
        nearby = sense_nearby!(agent, model)
        for prey in nearby
            if prey.species == :prey
                #print("moving to prey \n")
                direction = normalize(SVector(prey.pos...) .- agent.pos)
                if norm(direction) < speed
                    speed = norm(direction)
                end
            end
        end

    end

    if direction == [0.0, 0.0, 0.0]
        if rand() < RANDOM_WALK_CHANCE  # 20% chance of random walk
            agent.vel = normalize(SVector(randn(3)...))
        end
    else
        agent.vel = direction
    end

    # proposed_pos = ntuple(i -> clamp(agent.pos[i] .+ agent.vel[i] * speed, 0, EXTENT), 3)

    # # Boundary behavior: reflect direction if at boundary
    # newvel = Vector(agent.vel)
    # for i in 1:3
    #     if proposed_pos[i] <=0 || proposed_pos[i] >= 100
    #         newvel[i] *= -1  # Reverse velocity component
    #     end
    # end
    # agent.vel = SVector(newvel...)

    # new_pos = ntuple(i -> clamp(agent.pos[i] .+ agent.vel[i] * speed, 0., EXTENT), 3)

    new_pos = ntuple(i -> mod(agent.pos[i] + agent.vel[i] * speed, EXTENT), 3)

    move_agent!(agent, new_pos, model)

    move_agent!(agent, new_pos, model)

    # Lose energy based on speed and efficiency
    agent.energy -= speed^2 / agent.genome[3]

    # Handle reproduction
    if agent.energy >= REPRODUCTION_ENERGY_THRESHOLD  # Energy threshold for reproduction
        reproduce!(agent, model)
    end

    # Handle predation
    if agent.species == :predator
        hunt!(agent, model)
    end
end


function reproduce!(agent, model)
    agent.energy /= 2  # Split energy with offspring

    # Create offspring with mutated genome
    pos = ntuple(i -> clamp(agent.pos[i] + randn(), 0, EXTENT), 3)
    vel = Tuple(randn(3))
    new_genome = mutate_genome(agent.genome)

    add_agent!(
        pos,
        model,
        vel,
        agent.species,
        agent.energy,
        new_genome
    )
end

function hunt!(predator, model)
    nearby = sense_nearby!(predator, model)
    for prey in nearby
        if prey.species == :prey
            if norm(predator.pos - prey.pos) <= predator.genome[4]  # Within consumption range
                predator.energy += prey.energy  # Gain energy from prey
                remove_agent!(prey, model)
                break
            end
        end
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
        scatter!(ax, [resource[1]], [resource[2]], [resource[3]], color=get_resource_color(), markersize=10, marker=:hexagon, strokecolor=:white, strokewidth=1)
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
        scatter!(ax, x, y, z, marker=markers, color=colors, markersize=8, strokecolor=:white, strokewidth=1)
    end
    draw_resources!(resources)
    ax.title = "Step: $step"
end

# Set up visualization 
fig = Figure(size=(800, 800), backgroundcolor=:black)
ax = Axis3(fig[1, 1], aspect=:data, title="3D Ecosystem Simulation",
    xlabel="X", ylabel="Y", zlabel="Z",
    titlecolor=:white,
    xgridcolor=:white,
    ygridcolor=:white,
    zgridcolor=:white,
    backgroundcolor=:black,
    xspinecolor_1=:white,
    yspinecolor_1=:white,
    zspinecolor_1=:white,
    xspinecolor_2=:white,
    yspinecolor_2=:white,
    zspinecolor_2=:white,
    xspinecolor_3=:white,
    yspinecolor_3=:white,
    zspinecolor_3=:white,
)

function make_vid(fig)
    xlims!(ax, 0, EXTENT)
    ylims!(ax, 0, EXTENT)
    zlims!(ax, 0, EXTENT)

    # Record animation
    record(fig, "simulation.mp4", 1:N_STEPS, framerate=30,) do frame_number
        step!(model, agent_step!, model_step!)  # Simulate one step of the model
        update_plot!(ax, model, resources, frame_number)
        print("Step: ", frame_number, "\n")
        print("length(allagents(model)) = ", length(allagents(model)), "\n")
        print("length(resources) = ", length(resources), "\n")
    end
end

make_vid(fig)