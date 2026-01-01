# AIMM — Multi-Signal Suspicion Window Framework (Reference Implementation)

## Overview

This repository contains a lightweight reference implementation of the
Multi-Signal Suspicion Window (MSSW) framework described in our research paper:

"A Multi-Signal Suspicion Window Framework for Explainable Market Manipulation Detection."

This code is **illustrative**, not production-grade. We built it to show
how the framework can be implemented in practice, focusing on signal
aggregation, window-based analysis, and generating the figures used in the paper.

## What This Repo Does (and Doesn't Do)

This repository is **not**:
- A trading system or investment tool
- Providing market predictions or financial advice
- A complete or optimized detection system
- Benchmarking against other methods

What we're showing here:
- How to structure a multi-signal pipeline
- The concept of suspicion windows in action
- Combining different signals in an explainable way
- How we generated the figures in the paper

## Repository Structure

- `figures/`  
  Figures we generated for the paper, showing signal interactions and suspicion windows.

- `scripts/`  
  Python scripts that create the figures from sample data. We kept these simple
  on purpose—clarity over performance.

- `data/`  
  Small sample datasets for demonstration only. Everything here is either
  synthetic or anonymized.

- `README.md`  
  This file.

## Data

The data here is **demonstration only**. It's not real trading data from any
individual, institution, or proprietary source.

If anything looks like real market behavior, that's coincidental—we're just
using it for visualization.

## Reproducing Figures

You can use the scripts in `scripts/` to regenerate the figures from the paper.

We created these to:
- Show how signals interact
- Demonstrate suspicion windows emerging over time
- Support the qualitative analysis in the paper

Don't worry about exact numbers—we're focused on the overall patterns and structure.

## Relationship to the Paper

The paper is the main thing here. This repo just supplements it to help
with transparency and understanding.

Our conclusions don't depend on this code—the paper stands on its own.

## Limitations

We prioritized clarity over completeness. This means we've left out a lot of
things you'd need for real-world use: latency constraints, adversarial adaptation,
regulatory integration, large-scale optimization, and more.

We did this deliberately to keep the focus on the core framework.

## Disclaimer

This is for academic and educational purposes only. Nothing here is financial,
investment, or legal advice.

We're not responsible for any use of this code outside an academic or
educational setting.

## License

This project is released under the MIT License.

## Contact

Questions or comments related to the framework can be directed to the author
via the contact information provided in the associated paper.
