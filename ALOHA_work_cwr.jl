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
using PyCall

# installer gsw en Python (car avec Julia ne fonctionne pas)
#pyimport_conda("gsw", "gsw")  # ou pip install gsw depuis un terminal

gsw = pyimport("gsw")

# Variables I'm going to grid
DIC = aloha_filtered_df[!, :DIC_Umolkg]
ALK = aloha_filtered_df[!, :Alk_Umolkg]
SAL = aloha_filtered_df[!, :Salinity]
TMP = aloha_filtered_df[!, :Temperature]
LAT = aloha_filtered_df[!, :Latitude_N] |> float
LON = aloha_filtered_df[!, :Longitude_W] |> float
PRS = aloha_filtered_df[!, :Pressure_db]
NITR = aloha_filtered_df[!, :Nitr_Umolkg]
PHOS = aloha_filtered_df[!, :Phos_Umolkg]
SILI = aloha_filtered_df[!, :Sil_Umolkg]
# Calculate potential temperature
SA = gsw.SA_from_SP(SAL, PRS, LON, LAT) # Absolute Salinity
THETA = gsw.pt_from_t(SA, TMP, PRS, p_ref=0.0) # potential temperature


# Coordinates
TIME = decimal_year

DIC_mean = mean(DIC)
TMP_mean = mean(THETA) #was TMP before but changed to THETA
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

# These are the residuals from the line - we're gonna fit these now for each individual year
residuals = DIVAnd_residual(s, fi) # calulcates the residuals represent the differences between the observed data and the values predicted by the fitted model.

uniq_years = year.(date) |> unique

# Create a dictionary, with `yr` as the key and the corresponding residuals as the value
profiles = Dict{Int, Vector{Float64}}()

for yr in uniq_years
    # Get the the residuals for the year 1988
    yr_residuals = residuals[year.(date) .== yr]
    yr_prs = PRS[year.(date) .== yr]

    fi_residual, s_residual = DIVAndrun(
        mask,                  # 1D mask 
        pmn,                   # 1D resolution
        (pr_grid,),            # 1D grid (tuple with single element)
        (yr_prs,),                # 1D coordinates (tuple with single element)
        yr_residuals,              # Data to fit
        (lenz,),                # Correlation length for pressure only 
        0.1                    # Signal-to-noise ratio
    )

    profiles[yr] = TMP_mean .+ fi .+ fi_residual
end

p1 = Plots.plot(TMP_mean .+ fi, pr_grid, color=:black,linewidth=5, yflip=true)
for (yr, profile) in profiles
    # Plot the profile for each year
    Plots.plot!(profile, pr_grid, label=string(yr), alpha=0.5)
end

display(p1)

PRS = aloha_filtered_df[!, :Pressure_db]
TIME = decimal_year


function grid_variable(var)

    varmean = mean(var)
    varanom = var .- varmean

    pr_grid = collect(range(0.0,1100, step=10))

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


    lenz, _dbinfo = fitvertlen(
        (LON, LAT, PRS), 
        varanom, 
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
        varanom,               # Data to fit
        (lenz,),               # Correlation length for pressure only 
        0.1                    # Signal-to-noise ratio
    )


    residuals = DIVAnd_residual(s, fi)

    uniq_years = year.(date) |> unique

    # Create a dictionary, with `yr` as the key and the corresponding residuals as the value
    profiles = Dict{Int, Vector{Float64}}()

    for yr in uniq_years
        # Get the the residuals for the year 1988
        yr_residuals = residuals[year.(date) .== yr]
        yr_prs = PRS[year.(date) .== yr]

        fi_residual, s_residual = DIVAndrun(
            mask,                  # 1D mask 
            pmn,                   # 1D resolution
            (pr_grid,),            # 1D grid (tuple with single element)
            (yr_prs,),                # 1D coordinates (tuple with single element)
            yr_residuals,              # Data to fit
            (lenz,),                # Correlation length for pressure only 
            0.1                    # Signal-to-noise ratio
        )

        profiles[yr] = varmean .+ fi .+ fi_residual
    end

    return profiles
end

gridded_DIC = grid_variable(DIC)
gridded_SAL = grid_variable(SAL)
gridded_TMP = grid_variable(TMP)

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



function decompose_temp(init_temp::Vector{Float64},final_temp::Vector{Float64}
                       ,init_DIC::Vector{Float64},final_DIC::Vector{Float64}
                       ;transient_val::Float64=0.017
                       ,gaussian_width::Float64=0.0001)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}} # tuples have a fixed size and their elements can have different types
                    
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

    return excess_temp, redist_temp, κr

end

# Plot all the DIC profiles, with our initial profile bolded in black
p3 = Plots.plot(gridded_DIC[1988], pr_grid, color=:black,linewidth=5, yflip=true)
for (yr, profile) in gridded_DIC
    # Plot the profile for each year
    Plots.plot!(profile, pr_grid, label=string(yr), alpha=0.5, legend=false)
end

display(p3)


excess_temp_profiles = Dict{Float64, Vector{Float64}}()
redist_temp_profiles = Dict{Float64, Vector{Float64}}()
κr_profiles = Dict{Float64, Vector{Float64}}()

init_yr = uniq_years[1] #can be change if I want antohet inital year

for yr in uniq_years
    excess_temp_profiles[yr], redist_temp_profiles[yr],κr_profiles[yr] = decompose_temp(
        gridded_TMP[init_yr], gridded_TMP[yr],
        gridded_DIC[init_yr], gridded_DIC[yr],
        transient_val=0.017,
        gaussian_width=0.0001
    )
end


## Plot Temp excess versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
    profile_mat[:, i] = excess_temp_profiles[yr]
end

P4=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="ALOHA - Θe profiles over time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P4, joinpath(git_root, "data_output/ALOHA", "ALOHA_temperature_excess_time.png"))

## Plot Temp redistri versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
    profile_mat[:, i] = redist_temp_profiles[yr]
end

P5=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="Θredistri profiles over Time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P5, joinpath(git_root, "data_output/ALOHA", "ALOHA_temperature_redistri_time.png"))


## Plot Kr versus time as scatter
kr_profile = κr_profiles[uniq_years[1]] # pick juste one date
P8 = Plots.plot(kr_profile, pr_grid, color=:black,linewidth=5, yflip=true, legend = false, xlabel="κr_profile", ylabel="Pressure (db)", title="ALOHA - κr profile for year $(uniq_years[1])")
savefig(P8, joinpath(git_root, "data_output/ALOHA", "ALOHA_Kr_time.png"))

# Plot of the DIC-Temperature values
P6=Plots.scatter(DIC, TMP, zcolor=decimal_year, markersize=5, alpha=0.4, 
   legend=false, colorbar=true, xlabel="DIC", ylabel="Temperature",cmap = :jet1,)

savefig(P6, joinpath(git_root, "data_output/ALOHA", "ALOHA_temperature_DIC_time.png"))
# Plot of the DIC-Salinity values
P7=Plots.scatter(DIC, SAL, zcolor=decimal_year, markersize=5, alpha=0.4, 
   legend=false, colorbar=true, xlabel="DIC", ylabel="Salinity",cmap = :jet1,)

savefig(P7, joinpath(git_root, "data_output/ALOHA", "ALOHA_Salinity_DIC_time.png"))



# Save the data out
sorted_pairs = sort(collect(excess_temp_profiles), by=first)
sorted_columns = [Symbol(k)=>v for (k,v) in sorted_pairs]

DataFrame(;sorted_columns..., Pressure_db=pr_grid) |>
    CSV.write(joinpath(git_root, "output_files", "ALOHA_excess_temp_profiles.csv"), header=true)


sorted_pairs = sort(collect(redist_temp_profiles), by=first)
sorted_columns = [Symbol(k)=>v for (k,v) in sorted_pairs]

DataFrame(;[Symbol(k)=>v for (k,v) in redist_temp_profiles]..., Pressure_db=pr_grid) |>
    CSV.write(joinpath(git_root, "output_files", "ALOHA_redist_temp_profiles.csv"), header=true)




##### TEST with another Temp excess - redistri function
function ExcessRedistTempSalFromTempSalDIC(init_temp::Vector{Float64},final_temp::Vector{Float64}
    ,init_DIC::Vector{Float64},final_DIC::Vector{Float64}
    ,init_Sal::Vector{Float64},final_Sal::Vector{Float64}
    ;transient_val::Float64=0.017
    ,zDimension::Int64=1
    ,gaussian_width::Float64=0.0001)

if size(init_temp) == size(final_temp) == size(init_DIC) == size(final_DIC) ==
size(init_Sal) == size(final_Sal)
nothing
else
error("\"init_temp\", \"final_temp\", \"init_DIC\", \"final_DIC\"
\"init_Sal\", \"final_Sal\" must all be the same size")
end

ζ = central_diff(init_DIC) # DIC gradient
ξ = central_diff(init_temp)
η = central_diff(init_Sal)

κrT = ξ ./ ζ
κrS = η ./ ζ
τ = η ./ ξ

weight1 = gaussian(ζ,σ = gaussian_width)
weight2 = (weight1 .- 1.0) ./ (κrT .- transient_val)

LENGTH_1 = size(init_temp,1) # dimnension time
LENGTH_2 = size(init_temp,2) #dimension depth

mat1 = fill(NaN,3,2,LENGTH_1,LENGTH_2)
mat2 = fill(NaN,3,2,LENGTH_1,LENGTH_2)

for i = 1:LENGTH_1, j = 1:LENGTH_2
mat1[:,:,i,j] = [0              transient_val
;1              -transient_val
;τ[i,j]              -transient_val*τ[i,j]]

mat2[:,:,i,j] = [transient_val -transient_val*κrT[i,j]
;-κrT[i,j]         transient_val*κrT[i,j]
;-κrS[i,j]         transient_val*κrS[i,j] ]
end

ΔΘ = final_temp - init_temp
ΔDIC = final_DIC - init_DIC
ΔSal = final_Sal - init_Sal

inputVectors = fill(NaN,2,LENGTH_1,LENGTH_2);
outputVectors = fill(NaN,3,LENGTH_1,LENGTH_2);
inputVectors[1,:,:] = ΔΘ
inputVectors[2,:,:] = ΔDIC

for i =1:LENGTH_1, j = 1:LENGTH_2
outputVectors[:,i,j] = (weight1[i,j] * mat1[:,:,i,j] + weight2[i,j] * mat2[:,:,i,j]) * inputVectors[:,i,j]
end

ExcessTemperature = outputVectors[1,:,:]
RedistTemperature = outputVectors[2,:,:]
RedistSalinity    = outputVectors[3,:,:]
ExcessSalinity    = ΔSal - RedistSalinity

return ExcessTemperature, RedistTemperature, ExcessSalinity, RedistSalinity
end

excess_temp_profiles = Dict{Float64, Matrix{Float64}}()
redist_temp_profiles = Dict{Float64, Matrix{Float64}}()
excess_sal_profiles = Dict{Float64, Matrix{Float64}}()
redist_sal_profiles = Dict{Float64, Matrix{Float64}}()
init_date = uniq_years[1] # Change if a different init_date.

println("Using $init_date as the initial date")

for date in uniq_years
    excess_temp_profiles[date], 
    redist_temp_profiles[date], 
    excess_sal_profiles[date], 
    redist_sal_profiles[date] = ExcessRedistTempSalFromTempSalDIC(
        gridded_TMP[init_date], gridded_TMP[date],
        gridded_DIC[init_date], gridded_DIC[date],
        gridded_SAL[init_date], gridded_SAL[date];
        transient_val=0.017,
        gaussian_width=0.0001,
        zDimension=1
    )
end

## Plot Temp excess (OTHER METHOD) versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
    profile_mat[:, i] = excess_temp_profiles[yr]
end

P12=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="ALOHA - Θe profiles over time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P12, joinpath(git_root, "data_output/ALOHA", "ALOHA_temperature_excess_time_OTHER_FUNCTION.png"))
## Plot Temp excess (OTHER METHOD) versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
    profile_mat[:, i] = redist_temp_profiles[yr]
end

P13=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="ALOHA - Θredistri profiles over time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P13, joinpath(git_root, "data_output/ALOHA", "ALOHA_temperature_redistri_time_OTHER_FUNCTION.png"))

## Plot SAL excess (OTHER METHOD) versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
    profile_mat[:, i] = excess_sal_profiles[yr]
end

P14=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="ALOHA - SAL-e profiles over time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P14, joinpath(git_root, "data_output/ALOHA", "ALOHA_salinity_excess_time_OTHER_FUNCTION.png"))
## Plot SAL excess (OTHER METHOD) versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
    profile_mat[:, i] = redist_sal_profiles[yr]
end

P15=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="ALOHA - SAL-redistri profiles over time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P15, joinpath(git_root, "data_output/ALOHA", "ALOHA_salinity_redistri_time_OTHER_FUNCTION.png"))



function decompose_DIC(init_temp::Vector{Float64},final_temp::Vector{Float64}
    ,init_DIC::Vector{Float64},final_DIC::Vector{Float64}
    ;transient_val::Float64=0.017
    ,gaussian_width::Float64=0.0001)::Tuple{Vector{Float64}, Vector{Float64}} # tuples have a fixed size and their elements can have different types
 
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
weight2 = (weight1 .- 1.0) ./ (transient_val .-  κr )

mat1 = fill(NaN,2,2,VEC_LENGTH)
mat2 = fill(NaN,2,2,VEC_LENGTH)
mat3 = fill(NaN,2,2,VEC_LENGTH)

for i = 1:length(κr)
mat1[:,:,i] = [0               transient_val
;1              -transient_val]

mat2[:,:,i] = [-1   κr[i]
;1          -transient_val]

mat3[:,:,i] = [(1/transient_val) 0
;0              (1/κr[i])]
end

ΔΘ = final_temp - init_temp
ΔDIC = final_DIC - init_DIC

input_vec = fill(NaN,2,VEC_LENGTH); output_vec = copy(input_vec)
input_vec[1,:] = ΔΘ
input_vec[2,:] = ΔDIC

for i =1:VEC_LENGTH
output_vec[:,i] = (weight1[i] * (mat1[:,:,i] * mat3[:,:,i]) + weight2[i] * mat2[:,:,i]) * input_vec[:,i]
end

excess_DIC = output_vec[1,:]
redist_DIC = output_vec[2,:]

return excess_DIC, redist_DIC

end


excess_DIC_profiles = Dict{Float64, Vector{Float64}}()
redist_DIC_profiles = Dict{Float64, Vector{Float64}}()

init_yr = uniq_years[1] #can be change if I want antohet inital year

for yr in uniq_years
excess_DIC_profiles[yr], redist_DIC_profiles[yr] = decompose_DIC(
gridded_TMP[init_yr], gridded_TMP[yr],
gridded_DIC[init_yr], gridded_DIC[yr],
transient_val=0.017,
gaussian_width=0.0001
)
end


## Plot Temp excess versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
profile_mat[:, i] = excess_DIC_profiles[yr]
end

P4=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="ALOHA - DIC_excess profiles over time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P4, joinpath(git_root, "data_output/ALOHA", "ALOHA_DIC_excess_time.png"))

## Plot Temp redistri versus time as a heatmap
profile_mat = zeros(length(pr_grid), length(uniq_years))
for (i, yr) in enumerate(uniq_years)
profile_mat[:, i] = redist_DIC_profiles[yr]
end

P5=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="ALOHA - DIC redistri profiles over Time", yflip=true, colorbar=true, cmap=:bwr,)
#clim=(-0.5,0.5),)
Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)
savefig(P5, joinpath(git_root, "data_output/ALOHA", "ALOHA_DIC_redistri_time.png"))


## Determination of the pCO2 ocean usign CO2SYS 

using Pkg
# need to be in pkg mode i.e. ] to enter
#add https://github.com/mvdh7/CO2System.jl
using CO2System
par1type =    1 # The first parameter supplied is of type "1", which is "alkalinity"
par1     = ALK # value of the first parameter
par2type =    2 # The first parameter supplied is of type "1", which is "DIC"
par2     = DIC # value of the second parameter, which is a long vector of different DIC"s!
sal      =   SAL # Salinity of the sample
tempin   =   TMP # Temperature at input conditions
presin   =    PRS # Pressure    at input conditions
tempout  =    25 # Temperature at output conditions - doesn't matter in this example
presout  =    0 # Pressure    at output conditions - doesn't matter in this example
sil      =   1  # Concentration of silicate  in the sample (in umol/kg)
po4      =    1 # Concentration of phosphate in the sample (in umol/kg)
pHscale  =    1 # pH scale at which the input pH is reported ("1" means "Total Scale")  - doesn't matter in this example
k1k2c    =    4 # Choice of H2CO3 and HCO3- dissociation constants K1 and K2 ("4" means "Mehrbach refit")
kso4c    =    1 # Choice of HSO4- dissociation constants KSO4 ("1" means "Dickson")

A = CO2SYS(par1,par2,par1type,par2type,sal,tempin,tempout,presin,presout,
    sil,po4,pHscale,k1k2c,kso4c)[1]  ## [1] = data, [2] = header, [3] = nice header

    # The calculated pCO2's are in the 4th column of the output A of CO2SYS
sp1 = Plots.scatter(A[:,3], par2, color=:red, label="", ylabel="DIC", xlabel="pH", title="DIC vs pCO2",
marker=(:circle))


P7=Plots.scatter(A[:,4], par2, zcolor=decimal_year, markersize=5, alpha=0.4, 
   legend=false, colorbar=true, ylabel="DIC", xlabel="pCO2", title="DIC vs pCO2",cmap = :viridis,)

savefig(P7, joinpath(git_root, "data_output/ALOHA", "ALOHA_pCO2_DIC_time.png"))

#### Heatmap with the time as x axis and depth as y axis and pCO2 as color


uniq_years = unique(year.(date))
profile_mat = fill(NaN, length(pr_grid), length(uniq_years))

using Statistics, LinearAlgebra
using Interpolations

for (i, yr) in enumerate(uniq_years)
    idx = findall(year.(date) .== yr)
    vals = A[idx, 4]
    depths = A[idx, 43]

    if length(vals) < 20
        profile_mat[:, i] .= NaN
        continue
    end

    # tri par profondeur
    p = sortperm(depths)
    depths_sorted = depths[p]
    vals_sorted   = vals[p]

    # interpolation linéaire + extrapolation (NaN en dehors)
    itp = extrapolate(interpolate((depths_sorted,), vals_sorted, Gridded(Linear())), NaN)

    profile_mat[:, i] = [itp(d) for d in pr_grid]
end



P5=Plots.heatmap(uniq_years, pr_grid, profile_mat, color=:viridis, xlabel="Year", 
ylabel="Pressure (db)", title="Θredistri profiles over Time", yflip=true, ylim=(0,1000),colorbar=true,cmap=:bwr,
clim=(300,1200),)

#Plots.contour!(uniq_years, pr_grid, profile_mat, color=:black, alpha=0.5, levels=[0.0], linewidth=2)

 savefig(P5, joinpath(git_root, "data_output/ALOHA", "ALOHA_temperature_redistri_time.png"))