# Script to read the txt file and produce the decompition/excess tables
# and then save them in the data_output directory
print("bonjour Julia")

git_root = String(read(`git rev-parse --show-toplevel`))
git_root = strip(git_root)

# Creation of checksums for the data files in the data directory because
# that's the world we live in now....
using Glob
using JSON
using SHA

data_dir = joinpath(git_root, "data_input")
data_files = glob("*.txt", data_dir)


"""
Takes a path to a file and returns the SHA256 checksum of the file.
"""
function checksum(file_path::String)
    # This function takes a file path and returns the SHA256 checksum of the file.
    # It reads the file in chunks to avoid loading the entire file into memory.
    
    # Create a context for incremental hashing
    ctx = SHA.SHA256_CTX()
    
    open(file_path) do io
        while !eof(io)
            chunk = read(io, 65536)  # Read in chunks of 8192 bytes
            SHA.update!(ctx, chunk)
        end
    end
    
    # Finalize and get the digest
    return bytes2hex(SHA.digest!(ctx))
end

# Part of the script to update the checksums file
using JSON
checksums = Dict(basename(file) => checksum(joinpath(data_dir, file)) for file in data_files)
open(joinpath(git_root, "data_input", "checksums.json"), "w") do io
    JSON.print(io, checksums, 4)   # joli JSON indenté
end


"""
Loads the checksum file and returns a dictionary of checksums.
"""
function load_checksums()
    checksums_fname = joinpath(git_root, "data_input", "checksums.json")
    open(checksums_fname, "r") do IO
        return JSON.parse(IO)
    end
end

checksums = Dict{String, String}()

for file in data_files
    # Get the full path to the file
    file_path = joinpath(data_dir, file)
    
    # Calculate the checksum
    file_checksum = checksum(file_path)

    # Stick it in our dictionary
    checksums[basename(file)] = file_checksum
    
end

saved_checksums = load_checksums()

# Test they are the same - this will trip things up if the data files hav changed at all - even whitespace!
@assert checksums == saved_checksums "Checksums do not match! Please check the files."

# FIRST: BATS STATION
#import Pkg
#Pkg.add("DataFrames") # SAME for CSV
#to add the package if you don't have it already
using DataFrames
using CSV
using Pluto
#Pluto.run() --> open a Pluto notebook on your browser
# Load the dataset

bats_df = CSV.read(joinpath(git_root, "data_input", "BATS_Carbon_discrete_with_QF.txt"), DataFrame)


bats_df
# SOME CHECKS
using Plots
using Dates

# prend seulement la partie avant la virgule et enlève les espaces
clean_dates = strip.(first.(split.(bats_df[!, :Date], ",")))
date = Date.(clean_dates, DateFormat("mm/dd/yyyy"))

# Convert it to a decimal year ie. 2023.5 - needed to plot date as a colour
decimal_year = year.(date) .+ (month.(date) .- 1) ./ 12 .+ day.(date) ./ 365.25
p1 = scatter(bats_df[!, :Longitude_W], bats_df[!, :Latitude_N], zcolor=decimal_year, xlabel="Longitude", ylabel="Latitude", title="Scatter of Latitude and Longitude over Time", color=:viridis)
savefig(p1, joinpath(git_root, "data_output", "BATS_Lat_Long_scatter.png"))


#### PLOT to see where the datapoints are 
using CairoMakie
using GeoMakie
using GeoMakie.GeoJSON
using NaturalEarth

fig = Figure(size = (900, 600))

x_ticks = collect(range(-78, 0, step=2))  # Every 4 degrees from -90 to 0
y_ticks = collect(range(20, 40, step=1))  # Every 2 degrees from 20 to 45

# Create a GeoAxis with sensible limits
ax = GeoAxis(fig[1,1]; 
    # Set limits using the limits parameter instead
    limits = ((-85, -4), (20, 50)),  # (lon_min, lon_max), (lat_min, lat_max)
    xticks = x_ticks,
    yticks = y_ticks,
)

# Add scatter plot
GeoMakie.scatter!(ax, -bats_df[!, :Longitude_W], bats_df[!, :Latitude_N], 
    color=decimal_year, colormap=:viridis, 
    markersize=8)

# Add continents
poly!(ax, GeoMakie.land())

# Add title 
ax.title = "BATS obs locations in North Atlantic"

fig # Display the figure

# Narrowing of the dataset to constraint to the BATS region
n_obs = size(bats_df, 1)
println("There are $n_obs observations in the full dataset.")

filtered_df = bats_df[62 .< bats_df[!, :Longitude_W] .< 66, :]
filtered_df = filtered_df[31 .< filtered_df[!, :Latitude_N] .< 33, :]

println("There are $(size(filtered_df, 1)) observations in the filtered dataset prior to filtering for DIC.")

@info "Filtering all observations without a DIC value"

filtered_df = filter(row -> !isnan(row[:DIC_Umolkg]), filtered_df)

println("There are $(size(filtered_df, 1)) observations in the filtered dataset after filtering for DIC.")

filtered_df = filter(row -> row[:Pressure_db] < 500, filtered_df)

println("There are $(size(filtered_df, 1)) observations in the filtered dataset after filtering for DIC and the top 500m.")

clean_dates2 = strip.(first.(split.(filtered_df[!, :Date], ",")))
date = Date.(clean_dates2, DateFormat("mm/dd/yyyy"))
# Convert it to a decimal year ie. 2023.5 - needed to plot date as a colour
decimal_year = year.(date) .+ (month.(date) .- 1) ./ 12 .+ day.(date) ./ 365.25

p2 = scatter(decimal_year, filtered_df[!, :Depth_m], zcolor=filtered_df[!, :DIC_Umolkg], xlabel="Time", ylabel="Pressure [db]", yflip=true, legend= false, colorbar=true, clabel = "DIC [µmol/kg]",ylim = (0,800))
savefig(p2, joinpath(git_root, "data_output", "BATS_scatter_depth_DIC_TS.png"))

# Using of the earliest data as possible profile to construct our T-DIC curve
# Round the date and pool the data together 
uniq_dates = decimal_year |> unique |> sort

println("Elements two & three here are the same occupation - but they're not the exact same date")
println(uniq_dates[100:103])
rounded_uniq_dates = round.(uniq_dates * 26 ) / 26 
rounded_uniq_dates = round.(rounded_uniq_dates, digits = 3)
println("\nNow they are")
println(rounded_uniq_dates[100:103])
decimal_year = round.(decimal_year * 26) / 26
rounded_decimal_year = round.(decimal_year, digits = 3)

filtered_df[!, :decimal_year] = rounded_decimal_year
filtered_df

# --> this is the dataset that will be used for the rest of the analysis

using Statistics 
using DIVAnd

DIC = filtered_df[!, :DIC_Umolkg]
SAL = filtered_df[!, :Salinity]
TMP = filtered_df[!, :Temperature]
LAT = filtered_df[!, :Latitude_N] |> float
LON = filtered_df[!, :Longitude_W] |> float

# Coordinates
PRS = filtered_df[!, :Pressure_db]
TIME = filtered_df[!, :decimal_year]

DIC_mean = mean(DIC[.!isnan.(DIC)])
TMP_mean = mean(TMP[.!isnan.(TMP)])
SAL_mean = mean(SAL[.!isnan.(SAL)])


DIC_anom = DIC .- DIC_mean
TMP_anom = TMP .- TMP_mean
SAL_anom = SAL .- SAL_mean


uniq_dates = filtered_df[!, :decimal_year] |> unique |> sort

pr_grid = collect(range(0.0,500, step=10))

mask = fill(true,  length(pr_grid))


#########
function grid_variable(var)

    varmean = mean(var[.!isnan.(var)])
    varanom = var .- varmean

    pr_grid = collect(range(0.0,500, step=10))

    mask = fill(true,  length(pr_grid))

    #= 
    pmn is the inverse of the grid resolution - see https://gher-uliege.github.io/DIVAnd.jl/stable/#DIVAnd.DIVAndrun.
    Honestly I find the documentation really unhelpful, so I've just set it to 1 for now. 
    We'll fix it properly in the second fitting, which is more important anyway. Since our 
    observations are fairly well spaced, it's less of an issue than it might otherwise be.
    =#
    pmn = 1.0 .* (ones(size(mask)),) 

    """
    Takes a z value and returns a searchz value. This is used to determine the
    search distance for correlation length fitting. 
    """
    function search_z_func(z)
        if z < 750
            return 100
        else
            return 300
        end
    end


    # Create a fallback function that gives sensible correlation lengths based on depth
    function create_manual_lenz(z_grid)
        # Start with smaller correlation length at surface, gradually increasing with depth
        # These values can be tuned based on your knowledge of the system
        return 50.0 .+ 0.2 .* z_grid  # Linear increase with depth
    end

    lenz = create_manual_lenz(pr_grid)  # Default value

    # Try to fit correlation length, otherwise use manual function
    try
        lenz, _dbinfo = fitvertlen(
            (LON, LAT, PRS), 
            varanom, 
            pr_grid,
            searchz=search_z_func, 
            limitfun= (z, len) -> max(min(len, 300), 10)
        )
        @info "Successfully fit vertical correlation length"
    catch e
        @warn "Error in fitvertlen: $e"
        @info "Using manual correlation length function instead"
        # lenz is already set to the default value
    end

    @info "Performing initial (pooled data, background) DIVAnd fitting"
    fi, s = DIVAndrun(
        mask,                  # 1D mask 
        pmn,                   # 1D resolution
        (pr_grid,),            # 1D grid (tuple with single element)
        (PRS,),                # 1D coordinates (tuple with single element)
        varanom,               # Data to fit
        (lenz,),               # Correlation length for pressure only 
        0.1                    # Signal-to-noise ratio
    )


    residuals = DIVAnd_residual(s, fi)


    # Create a dictionary, with `yr` as the key and the corresponding residuals as the value
    profiles = Dict{Float64, Vector{Float64}}()

    for date in uniq_dates
        # Get the the residuals for the year 1988
        dt_residuals = residuals[TIME .== date]
        dt_prs = PRS[TIME .== date]

        fi_residual, s_residual = DIVAndrun(
            mask,                  # 1D mask 
            pmn,                   # 1D resolution
            (pr_grid,),            # 1D grid (tuple with single element)
            (dt_prs,),                # 1D coordinates (tuple with single element)
            dt_residuals,              # Data to fit
            (lenz,),                # Correlation length for pressure only 
            0.1                    # Signal-to-noise ratio
        )

        profiles[date] = varmean .+ fi .+ fi_residual
    end

    return profiles
end

# GRID grid_variable

gridded_DIC = grid_variable(DIC)
gridded_SAL = grid_variable(SAL)
gridded_TMP = grid_variable(TMP)

# decomposition

using LinearAlgebra

"""
Takes the central difference of a vector.
"""
function central_diff(v::AbstractVector{<:Real})::AbstractVector{<:Real}
    # Very simple central difference funciton
    dv_fwds  = diff(v)
    dv_bwds  = reverse(-diff(reverse(v)))
    dx = Vector{AbstractFloat}(undef,length(v))
    dx[1] = dv_fwds[1]; dx[end] = dv_bwds[end]
    dx[2:end-1] = (dv_fwds[1:end-1] + dv_bwds[2:end]) / 2
    return dx
end

function gaussian(x;μ::Float64=0.0,σ::Float64)
    if ndims(x) > 0
      μ = fill(μ,size(x))
      σ = fill(σ,size(x))
    end
  
    gOfX = exp.( (-1/2) .* ( (x - μ) ./ σ).^2)
  
    return gOfX
  end

# This is stolen directly out of my old toolbox - it could do with a little love,
# hopefully you can follow what it's doing. All those other comments about me coming
# back to clean up code apply here too.
function decompose_temp(init_temp::Vector{Float64},final_temp::Vector{Float64}
                       ,init_DIC::Vector{Float64},final_DIC::Vector{Float64}
                       ;transient_val::Float64=0.017
                       ,gaussian_width::Float64=0.0001)::Tuple{Vector{Float64}, Vector{Float64}}
                    
    #= 
    The gaussain width is used to switch out the more complicated decomposition 
    (mat2) with the simpler one (mat1) when the DIC gradient is small. This is 
    what I got from cancelling the infinities out on paper - they wont on a computer 
    unfortunately, so it has to be done by hand.
    =#

    @assert size(init_temp) == size(final_temp) &&
           size(init_DIC) == size(final_DIC) &&
           size(init_temp) == size(init_DIC) "All input vectors (\"init_temp\", \"final_temp\", \"init_DIC\", \"final_DIC\") must have the same size"

    VEC_LENGTH::Int64 = length(init_temp)

    ζ = central_diff(init_DIC)
    ξ = central_diff(init_temp)

    κr = ξ ./ ζ



    weight1 = gaussian(ζ,σ = gaussian_width)
    weight2 = (weight1 .- 1.0) ./ (κr .- transient_val)

    mat1 = fill(NaN,2,2,VEC_LENGTH)
    mat2 = fill(NaN,2,2,VEC_LENGTH)

    for i = 1:length(κr)
    mat1[:,:,i] = [0               transient_val
                  ;1              -transient_val]

    mat2[:,:,i] = [transient_val -transient_val*κr[i]
                  ;-κr[i]          transient_val*κr[i]]
    end

    ΔΘ = final_temp - init_temp
    ΔDIC = final_DIC - init_DIC

    input_vec = fill(NaN,2,VEC_LENGTH); output_vec = copy(input_vec)
    input_vec[1,:] = ΔΘ
    input_vec[2,:] = ΔDIC

    for i =1:VEC_LENGTH
        output_vec[:,i] = (weight1[i] * mat1[:,:,i] + weight2[i] * mat2[:,:,i]) * input_vec[:,i]
    end

    excess_temp = output_vec[1,:]
    redist_temp = output_vec[2,:]

    return excess_temp, redist_temp

end

# Now we loop over the years and do the decomposition for each year. Start with 
# the first as both init and final as a sanity check for all zeros

excess_temp_profiles = Dict{Float64, Vector{Float64}}()
redist_temp_profiles = Dict{Float64, Vector{Float64}}()

init_date = uniq_dates[101] # Change this if you want to use a different init_date.
# Currently, set to the last date in 1996
println("Using $init_date as the initial date")

for date in uniq_dates
    excess_temp_profiles[date], redist_temp_profiles[date] = decompose_temp(
        gridded_TMP[init_date], gridded_TMP[date],
        gridded_DIC[init_date], gridded_DIC[date],
        transient_val=0.017,
        gaussian_width=0.0001
    )
end

# Plot all the DIC profiles, with our initial profile bolded in black
p3 = Plots.plot(gridded_DIC[1996.962], pr_grid, color=:black,linewidth=5, yflip=true)
for (yr, profile) in gridded_DIC
    # Plot the profile for each year
    Plots.plot!(profile, pr_grid, label=string(yr), alpha=0.5, legend=false)
end

display(p3)

## Plot Temp excess versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_dates))
for (i, yr) in enumerate(uniq_dates)
    profile_mat[:, i] = excess_temp_profiles[yr]
end

P4=Plots.heatmap(uniq_dates, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="BATS - Θe profiles over time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_dates, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P4, joinpath(git_root, "data_output", "BATS_temperature_excess_time.png"))

## Plot Temp redistri versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_dates))
for (i, yr) in enumerate(uniq_dates)
    profile_mat[:, i] = excess_temp_profiles[yr]
end

P4=Plots.heatmap(uniq_dates, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="Θe profiles over Time", yflip=true, colorbar=true, cmap=:bwr,
clim=(-0.5,0.5),)
Plots.contour!(uniq_dates, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
