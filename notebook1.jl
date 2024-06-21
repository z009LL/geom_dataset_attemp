### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ cd400bf3-672a-46c8-be7a-11e6c0db5ec2
using JSON


# ╔═╡ c5f5a579-b6c0-4d45-88b3-773016d314bf
function load_json(file_path)
    open(file_path, "r") do file
        return JSON.parse(file)
    end
end

# ╔═╡ b00031c1-57b8-4e03-918d-2ca66783dcb2
begin
	base_path = "C:\\Users\\Felipe\\Documents\\003_EXPERIMENTAL_003\\ARC\\"
	training_challenges = load_json(base_path * "arc-agi_training_challenges.json")
	training_solutions = load_json(base_path * "arc-agi_training_solutions.json")
	# evaluation_challenges = load_json(base_path * "arc-agi_evaluation_challenges.json")
	# evaluation_solutions = load_json(base_path * "arc-agi_evaluation_solutions.json")
end

# ╔═╡ 26ba2a52-84a8-4464-9fa9-2a395c2d48c4
function recall(i, j)
    k = sort(collect(keys(training_challenges)))
    t = k[i] 
    task = training_challenges[t]
    println("Set #", i, ", ", t)
    println("Input:")
    display(task["train"][j]["input"])  # Ajustar el índice para Julia
    println("Output:")
    display(task["train"][j]["output"]) # Ajustar el índice para Julia
end

# ╔═╡ dfe22fb7-0a93-471f-a245-5a7bc46f5569
recall(5, 1)

# ╔═╡ 7ca2db55-2864-4f06-a2e6-764610b160f1


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"

[compat]
JSON = "~0.21.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "ee04d9f7eb5ab00d856f3a8a382fc7b5194aba2f"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╠═cd400bf3-672a-46c8-be7a-11e6c0db5ec2
# ╠═c5f5a579-b6c0-4d45-88b3-773016d314bf
# ╠═b00031c1-57b8-4e03-918d-2ca66783dcb2
# ╠═26ba2a52-84a8-4464-9fa9-2a395c2d48c4
# ╠═dfe22fb7-0a93-471f-a245-5a7bc46f5569
# ╠═7ca2db55-2864-4f06-a2e6-764610b160f1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
