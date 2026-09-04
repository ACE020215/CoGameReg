# Benchmark data

Dataset files are not redistributed in this repository. The ten ARFF files used
by `run_benchmark.py` were copied without modification from the public data
directory of the official S2RMS repository:

- Repository: <https://github.com/BetaCatPro/S2RMS>
- Source snapshot: `316fb8159cfbbae53d316d78af0c2981ba616476`
- Paper: L. Liu, J. Zhang, K. Qian, and F. Min, "Semi-supervised
  regression via embedding space mapping and pseudo-label smearing,"
  *Applied Intelligence*, vol. 54, pp. 9622--9640, 2024.
- DOI: <https://doi.org/10.1007/s10489-024-05686-6>

According to Table 2 of the S2RMS paper, the original sources are:

| File | Dataset | Original source |
|---|---|---|
| `abalone.arff` | Abalone | UCI |
| `bank32nh.arff` | Bank32nh | Delve |
| `elevators.arff` | Elevators | UCI |
| `Folds5x2_pp.arff` | Folds5x2_pp | UCI |
| `kin8nm.arff` | Kin8nm | Delve |
| `parkinsons.arff` | Parkinsons | UCI |
| `puma8NH.arff` | Puma8NH | UCI |
| `space_ga.arff` | Space_ga | StatLib |
| `wind.arff` | Wind | StatLib |
| `wine_quality.arff` | Wine_quality | UCI |

## Preparation

1. Download or clone the official S2RMS repository.
2. Use the source snapshot identified above when available.
3. Copy the ten listed files from `S2RMS/data/` into this directory.
4. From the CoGameReg repository root, run:

   ```bash
   python scripts/verify_data.py
   ```

Successful verification prints `OK` for all ten files. The expected hashes in
`SHA256SUMS` were calculated from the S2RMS archive used for the reported
experiments.

The original providers and S2RMS authors retain their respective rights. This
repository does not grant a license for third-party datasets. Users are
responsible for consulting the original dataset terms.
