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


# Part of the script to update the checksums file
using JSON
checksums = Dict(basename(file) => checksum(joinpath(data_dir, file)) for file in data_files)
open(joinpath(git_root, "data_input", "checksums.json"), "w") do io
    JSON.print(io, checksums, 4)   # joli JSON indenté
end


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
Pluto.run()
# Load the dataset

bats_df = CSV.read(joinpath(git_root, "data_input", "BATS_Carbon_discrete_with_QF.txt"), DataFrame)


bats_df
# Let's just do a couple of minor checks that we haven't moved around too much.
using Plots
using Dates

# prend seulement la partie avant la virgule et enlève les espaces
clean_dates = strip.(first.(split.(bats_df[!, :Date], ",")))
date = Date.(clean_dates, DateFormat("mm/dd/yyyy"))

# Convert it to a decimal year ie. 2023.5 - needed to plot date as a colour
decimal_year = year.(date) .+ (month.(date) .- 1) ./ 12 .+ day.(date) ./ 365.25
p1 = scatter(bats_df[!, :Longitude_W], bats_df[!, :Latitude_N], zcolor=decimal_year, xlabel="Longitude", ylabel="Latitude", title="Scatter of Latitude and Longitude over Time", color=:viridis)
