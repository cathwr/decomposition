print("bonjour Julia")

git_root = String(read(`git rev-parse --show-toplevel`))
git_root = strip(git_root)

# I'm going to create checksums for the data files in the data directory because
# that's the world we live in now....
using Glob
using JSON
using SHA

data_dir = joinpath(git_root, "data")
data_files = glob("*.txt", data_dir)
