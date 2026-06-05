# Confidentiality and publication boundary

This repository is intended as a public research-software and portfolio companion to a published paper and public Zenodo dataset.

It does **not** include:

- raw FastBlade audio recordings;
- private facility operation data;
- confidential test logs;
- proprietary control logic;
- partner data;
- private reports;
- large generated WST arrays;
- generated intermediate feature-map arrays.

The workflow expects processed WST `.npy` arrays that are already publicly released on Zenodo.

Any extension of this repository with additional data should first check:

1. whether the data are public;
2. whether they have a clear license;
3. whether they expose facility, partner or industrial information;
4. whether they are small enough for Git or should be hosted as a release/data record.
