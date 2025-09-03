# Script to read the txt file and produce the decompition/excess tables --  ALOHA station
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


############################ The above part is similar in all the work files ############################
# ALOHA STATION
#import Pkg
#Pkg.add("DataFrames") # SAME for CSV
#to add the package if you don't have it already
using DataFrames
using CSV
using Pluto
#Pluto.run() --> open a Pluto notebook on your browser
# Load the dataset

aloha_df = CSV.read(joinpath(git_root, "data_input", "ALOHA_Carbon_discrete.txt"), DataFrame)

using Dates
using Plots

# prend seulement la partie avant la virgule et enlève les espaces

aloha_filtered_df = filter(row -> !isnan(row[:DIC_Umolkg]), aloha_df) #remove the lines when DIC = NaN
aloha_filtered_df = filter(row -> row[:Pressure_db] < 1100, aloha_filtered_df) #Ge3t rid of anything deeper than about a thousand metres - makes fitting correlation 
aloha_filtered_df = filter(row -> row[:Date] != "NaN/NaN/NaN", aloha_filtered_df) # Some of the dates are "NaN/NaN/NaN" - so we need to filter those out
println("There are $(size(aloha_filtered_df, 1)) observations in the filtered dataset after filtering for DIC & depth.")


clean_dates = strip.(first.(split.(aloha_filtered_df[!, :Date], ",")))
date = Date.(clean_dates, DateFormat("mm/dd/yyyy"))

# Convert it to a decimal year ie. 2023.5 - needed to plot date as a colour
decimal_year = year.(date) .+ (month.(date) .- 1) ./ 12 .+ day.(date) ./ 365.25
p1 = scatter(aloha_filtered_df[!, :Longitude_W], aloha_filtered_df[!, :Latitude_N], zcolor=decimal_year, xlabel="Longitude", ylabel="Latitude", title="Scatter of Latitude and Longitude over Time", color=:viridis, legend= false)
savefig(p1, joinpath(git_root, "data_output/ALOHA", "ALOHA_Lat_Long_scatter.png"))

# Plot showing where the observations are located in the Pacific ocean 
using CairoMakie
using GeoMakie
using GeoMakie.GeoJSON
using NaturalEarth

fig2 = Figure(size = (600, 600))

x_ticks = collect(range(-170, -145, step=2))  # Every 4 degrees from -90 to 0
y_ticks = collect(range(01, 30, step=1))  # Every 2 degrees from 20 to 45

# Create a GeoAxis with sensible limits
ax = GeoAxis(fig2[1,1]; 
    # Set limits using the limits parameter instead
    limits = ((-170, -145), (10, 40)),  # (lon_min, lon_max), (lat_min, lat_max)
    xticks = x_ticks,
    yticks = y_ticks,
)

# Add scatter plot
# Use filtered_df instead of aloha_df
GeoMakie.scatter!(ax, aloha_filtered_df[!, :Longitude_W], aloha_filtered_df[!, :Latitude_N], 
    color=decimal_year, colormap=:viridis, 
    markersize=10)

# Add continents
poly!(ax, GeoMakie.land())

# Add title 
ax.title = "ALOHA observation locations - Pacific"

fig2
#savefig(fig2, joinpath(git_root, "data_output", "ALOHA_Lat_Long_scatter.png"))
save(joinpath(git_root, "data_output/ALOHA", "ALOHA_Lat_Long_scatter.png"), fig2) #the savefig fucntion doesn't work in makie (geomakie)


p2 = scatter(decimal_year, aloha_filtered_df[!, :Pressure_db], zcolor=aloha_filtered_df[!, :Temperature], xlabel="Time", ylabel="Pressure [db]", yflip=true, legend= false, colorbar=true, clabel = "Temperature [ºC]",ylim = (0,800))
savefig(p2, joinpath(git_root, "data_output/ALOHA", "ALOHA_scatter_depth_Temp_TS.png"))

p2 = scatter(decimal_year, aloha_filtered_df[!, :Pressure_db], zcolor=aloha_filtered_df[!, :DIC_Umolkg], xlabel="Time", ylabel="Pressure [db]", yflip=true, legend= false, colorbar=true, clabel = "DIC [µmol/kg]",ylim = (0,800))
savefig(p2, joinpath(git_root, "data_output/ALOHA", "ALOHA_scatter_depth_DIC_TS.png"))


decimal_year |> unique |> sort #The piping operator (|>) passes the result of one expression as the first argument to the next expression.
#here first the date are made unique and then sorted

using Statistics 
using DIVAnd
# Variables I'm going to grid - just add more variables if you want to grid them
DIC = aloha_filtered_df[!, :DIC_Umolkg]
SAL = aloha_filtered_df[!, :Salinity]
TMP = aloha_filtered_df[!, :Temperature]
LAT = aloha_filtered_df[!, :Latitude_N] |> float
LON = aloha_filtered_df[!, :Longitude_W] |> float

# Coordinates
PRS = aloha_filtered_df[!, :Pressure_db]
TIME = decimal_year

DIC_mean = mean(DIC)
TMP_mean = mean(TMP)
SAL_mean = mean(SAL)


DIC_anom = DIC .- DIC_mean
TMP_anom = TMP .- TMP_mean
SAL_anom = SAL .- SAL_mean

pr_grid = collect(range(0.0,1100, step=10))

mask = fill(true,  length(pr_grid))

#= 
pmn is the inverse of the grid resolution - see https://gher-uliege.github.io/DIVAnd.jl/stable/#DIVAnd.DIVAndrun.
Just set it to 1 for now:  Since observations are fairly well spaced, it's less of an issue than it might otherwise be.
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


lenz, _dbinfo = fitvertlen(
    (LON, LAT, PRS), 
    TMP_anom, 
    pr_grid,
    searchz=search_z_func, 
    limitfun= (z, len) -> max(min(len, 300), 10)
)

@info "Performing initial (pooled data, background) DIVAnd fitting"
fi, s = DIVAndrun(
    mask,                  # 1D mask 
    pmn,                   # 1D resolution
    (pr_grid,),            # 1D grid (tuple with single element)
    (PRS,),                # 1D coordinates (tuple with single element)
    TMP_anom,              # Data to fit
    (lenz,),                # Correlation length for pressure only 
    0.1                    # Signal-to-noise ratio
)

Plots.scatter(TMP_mean .+ TMP_anom, PRS, yflip=true)
Plots.plot!(TMP_mean .+ fi, pr_grid, color=:black,linewidth=3) # Plot the gridded data on top of the scatter

