using Documenter
using NeuralOT

makedocs(
    sitename = "NeuralOT.jl",
    modules  = [NeuralOT],
    authors  = "NeuralOT.jl contributors",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://YOUR_USERNAME.github.io/NeuralOT.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Methods" => [
            "Dual OT (Seguy)"        => "methods/dual.md",
            "W2 via ICNN (Makkuva)"  => "methods/w2.md",
            "Flow matching"          => "methods/flow.md",
        ],
        "Evaluation" => "evaluation.md",
        "Troubleshooting" => "troubleshooting.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/YOUR_USERNAME/NeuralOT.jl.git",
    devbranch = "main",
)
