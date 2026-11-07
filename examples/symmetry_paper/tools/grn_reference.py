"""Reference fixtures for the GRN port (tests/data/grn_reference.json).

Runs the original R2 GRN code of Miras and Feenstra (2026) unchanged:
geno_body_GRN_v3.py, _mutate_v1.py (mutate_body) and _crossover_v1.py from
karinemiras/revolve2, branch plasticoding_v3 (commit 593bfc2). Development uses
only the revolve2 module classes' slots, so revolve2's Core, Brick, ActiveHinge
and Module are loaded from the checkout and Body is stubbed (a core and a no-op
finalize); nothing else of revolve2 is needed.

Usage (from the repo root):
    git clone -b plasticoding_v3 https://github.com/karinemiras/revolve2 /tmp/revolve2
    python -I examples/symmetry_paper/tools/grn_reference.py /tmp/revolve2 \\
        -o examples/symmetry_paper/tests/data/grn_reference.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import types
from pathlib import Path

MAX_MODULES = 26  # the core and 25 modules
N_GENOMES = 60


def load_original(repo: Path):
    mr = repo / "core" / "revolve2" / "core" / "modular_robot"
    cpp = repo / "genotypes" / "cppnwin" / "revolve2" / "genotypes" / "cppnwin"
    pkg = types.ModuleType("modular_robot")
    pkg.__path__ = []
    sys.modules["mr_pkg"] = pkg
    ns: dict = {}
    for name in ("_not_finalized_error", "_module", "_core", "_brick", "_active_hinge"):
        mod = types.ModuleType(f"mr_pkg.{name}")
        mod.__package__ = "mr_pkg"
        sys.modules[f"mr_pkg.{name}"] = mod
        exec(compile((mr / f"{name}.py").read_text(), str(mr / f"{name}.py"), "exec"), mod.__dict__)
        ns.update({k: v for k, v in mod.__dict__.items() if not k.startswith("__")})

    class Body:
        def __init__(self):
            self.core = ns["Core"](0.0)

        def finalize(self):
            pass

    class Genotype:
        def __init__(self, genotype):
            self.genotype = genotype

    fake = types.ModuleType("revolve2.core.modular_robot")
    fake.ActiveHinge, fake.Body, fake.Brick, fake.Core, fake.Module = (
        ns["ActiveHinge"], Body, ns["Brick"], ns["Core"], ns["Module"])
    sys.modules["revolve2"] = types.ModuleType("revolve2")
    sys.modules["revolve2.core"] = types.ModuleType("revolve2.core")
    sys.modules["revolve2.core.modular_robot"] = fake
    gen_mod = types.ModuleType("grn_pkg._genotype")
    gen_mod.Genotype = Genotype
    sys.modules["grn_pkg"] = types.ModuleType("grn_pkg")
    sys.modules["grn_pkg"].__path__ = []
    sys.modules["grn_pkg._genotype"] = gen_mod
    sys.modules["grn_pkg.modular_robot"] = types.ModuleType("grn_pkg.modular_robot")
    sys.modules["grn_pkg.modular_robot"].__path__ = []

    def load(path: Path, name: str, src_filter=lambda s: s):
        mod = types.ModuleType(name)
        mod.__package__ = name.rsplit(".", 1)[0]
        sys.modules[name] = mod
        exec(compile(src_filter(path.read_text()), str(path), "exec"), mod.__dict__)
        return mod

    grn = load(cpp / "modular_robot" / "geno_body_GRN_v3.py", "grn_pkg.modular_robot.geno_body_GRN_v3")
    # _mutate_v1 imports multineat for the brain mutation only
    mut = load(cpp / "_mutate_v1.py", "grn_pkg._mutate_v1", lambda s: s.replace("import multineat\n", ""))
    cx = load(cpp / "_crossover_v1.py", "grn_pkg._crossover_v1")
    return grn, mut.mutate_body, cx.crossover_v1, Genotype, ns


def body_record(grn_mod, Genotype, ns, values: list[float]) -> dict:
    g = grn_mod.GRN(MAX_MODULES, "reg2m2", Genotype(list(values)), 0, None, 1, 0)
    try:
        body, _ = g.develop()
    except IndexError:  # no genes: the original cannot inject (maternal_injection)
        return {"error": "no genes"}
    modules, queue = [], [(body.core, None, None)]
    names = {ns["Core"]: "CORE", ns["Brick"]: "BRICK", ns["ActiveHinge"]: "HINGE"}
    while queue:
        m, parent, slot = queue.pop(0)
        modules.append({"id": str(m._id), "parent": parent, "slot": slot, "type": names[type(m)],
                        "rotation": round(m.rotation / (3.141592653589793 / 2)),
                        "absolute_rotation": getattr(m, "_absolute_rotation", 0),
                        "coords": list(m.substrate_coordinates)})
        queue += [(c, str(m._id), i) for i, c in enumerate(m.children) if c is not None]
    return {"modules": sorted(modules, key=lambda r: int(r["id"])), "n_genes": len(g.genes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("revolve2", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    grn_mod, mutate_body, crossover_v1, Genotype, ns = load_original(args.revolve2)

    rng = random.Random(2026)
    genomes = [grn_mod.random_v1(rng).genotype for _ in range(N_GENOMES)]
    # longer and recombined genomes too
    for k in range(20):
        a, b = genomes[k], genomes[k + 20]
        genomes.append(crossover_v1(Genotype(list(a)), Genotype(list(b)), random.Random(k)).genotype)

    develop = [{"genome": g, **body_record(grn_mod, Genotype, ns, g)} for g in genomes]
    mutations = []
    for k, g in enumerate(genomes[:40]):
        seed = 100 + k
        mutations.append({"genome": g, "seed": seed,
                          "child": mutate_body(Genotype(list(g)), random.Random(seed)).genotype})
    crossovers = []
    for k in range(30):
        a, b, seed = genomes[k], genomes[(k * 7 + 3) % len(genomes)], 500 + k
        crossovers.append({"a": a, "b": b, "seed": seed,
                           "child": crossover_v1(Genotype(list(a)), Genotype(list(b)), random.Random(seed)).genotype})
    commit = subprocess.run(["git", "-C", str(args.revolve2), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source": f"karinemiras/revolve2@{commit} (plasticoding_v3)",
                                    "max_modules": MAX_MODULES, "tfs": "reg2m2",
                                    "develop": develop, "mutate": mutations, "crossover": crossovers}))
    sizes = [len(d.get("modules", [])) for d in develop]
    print(f"{len(develop)} bodies (modules incl. core: min {min(sizes)}, mean {sum(sizes)/len(sizes):.1f}, "
          f"max {max(sizes)}), {len(mutations)} mutations, {len(crossovers)} crossovers -> {args.out}")


if __name__ == "__main__":
    main()
