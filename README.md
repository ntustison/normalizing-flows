# normflows (fork)

A PyTorch package for discrete normalizing flows (NICE, Real NVP, Glow, Neural Spline Flows, etc.).
This repository is a **fork** of the original project by Vincent Stimper and collaborators at
https://github.com/VincentStimper/normalizing-flows. It keeps the import path **`normflows`** and
adapts the project to a modern `src/` layout with a simplified test and packaging setup.

> If you are looking for the canonical upstream documentation and examples, please refer to the
> original project’s site: https://vincentstimper.github.io/normalizing-flows/

## Installation

Requires Python ≥ 3.10 and a working PyTorch installation (GPU optional).

```bash
pip install -e .
```

To run example notebooks in this fork (if you keep them), install extras:

```bash
pip install -e .[examples]
```

## Quick start

```python
import normflows as nf

# Base distribution (2D diagonal Gaussian)
base = nf.distributions.base.DiagGaussian(2)

# Real NVP with simple MLP conditioner
flows = []
num_layers = 8
for _ in range(num_layers):
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(2, mode="swap"))

model = nf.NormalizingFlow(base, flows)
loss = model.forward_kld(x)  # x: (batch, 2)
loss.backward()
```

## Documentation

For in-depth API docs and background, use the upstream documentation:
https://vincentstimper.github.io/normalizing-flows/

## Attribution

This is a derivative work based on the upstream project:
- **Original repository:** https://github.com/VincentStimper/normalizing-flows
- **License:** MIT (see `LICENSE` in this repository and upstream)
- **Upstream authors:** Vincent Stimper et al.

This fork focuses on:
- `src/`/`tests/` restructure for packaging and CI.
- Modern `pyproject.toml` build config.
- Stability options in Glow-style flows (e.g., optional `tanh`-capped scales, gradient clipping, sentry checks).

## Citation

If you use `normflows`, please cite the corresponding paper:

> Stimper et al., (2023). normflows: A PyTorch Package for Normalizing Flows.
> Journal of Open Source Software, 8(86), 5361, https://doi.org/10.21105/joss.05361

**BibTeX**

```bibtex
@article{Stimper2023,
  author = {Vincent Stimper and David Liu and Andrew Campbell and Vincent Berenz and Lukas Ryll and Bernhard Schölkopf and José Miguel Hernández-Lobato},
  title = {normflows: A PyTorch Package for Normalizing Flows},
  journal = {Journal of Open Source Software},
  volume = {8},
  number = {86},
  pages = {5361},
  publisher = {The Open Journal},
  doi = {10.21105/joss.05361},
  url = {https://doi.org/10.21105/joss.05361},
  year = {2023}
}
```

## Contributing

Pull requests are welcome. Please add tests for any new functionality and run:

```bash
pytest -q
```
