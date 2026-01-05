# About ANDES

ANDES is an open-source Python library for power system modeling, computation, analysis, and control.

## Key Features

- **Power flow** calculation using Newton-Raphson method
- **Time-domain simulation** (transient stability) with implicit trapezoidal integration
- **Eigenvalue analysis** (small-signal stability) with participation factor computation
- **Symbolic-numeric framework** for rapid model prototyping
- **Full second-generation renewable energy models** following WECC specifications

## Design Philosophy

ANDES uses a hybrid symbolic-numeric approach where:

1. Device models are defined symbolically using Python classes
2. Equations are processed by SymPy to generate optimized numerical code
3. Numerical simulation uses the generated code for performance

This approach enables:

- Rapid prototyping of new device models
- Automatic Jacobian derivation
- Self-documenting models with LaTeX equation export

## Citation

If you use ANDES in your research, please cite:

> H. Cui, F. Li and K. Tomsovic, "Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis," *IEEE Transactions on Power Systems*, vol. 36, no. 2, pp. 1373-1384, March 2021. [DOI: 10.1109/TPWRS.2020.3017019](https://doi.org/10.1109/TPWRS.2020.3017019)

BibTeX entry:

```bibtex
@article{cui2021hybrid,
  author={Cui, Hantao and Li, Fangxing and Tomsovic, Kevin},
  journal={IEEE Transactions on Power Systems},
  title={Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis},
  year={2021},
  volume={36},
  number={2},
  pages={1373-1384},
  doi={10.1109/TPWRS.2020.3017019}
}
```

## Acknowledgments

ANDES is developed at the [CURENT](https://curent.utk.edu/) research center, a National Science Foundation Engineering Research Center for Ultra-Wide-Area Resilient Electric Energy Transmission Networks.

## License

ANDES is released under the GPL-3.0 license. See the [LICENSE](https://github.com/CURENT/andes/blob/master/LICENSE) file for details.

## Links

- **Source Code**: [github.com/CURENT/andes](https://github.com/CURENT/andes)
- **Documentation**: [docs.andes.app](https://docs.andes.app)
- **PyPI**: [pypi.org/project/andes](https://pypi.org/project/andes/)
- **Issue Tracker**: [GitHub Issues](https://github.com/CURENT/andes/issues)
