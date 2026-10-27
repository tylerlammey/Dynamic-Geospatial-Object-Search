# Third-party sources and model terms

This standalone benchmark calls existing project helpers and native PyTorch,
Transformers, timm, PEFT, Accelerate, NumPy, Pillow and geospatial libraries.
Their respective licenses remain applicable; no runtime package is copied into
tracked source. The ignored dependency overlay is a local installation.

The D3 adapter's prompt compilation, attention contract and flattened patch
projection follow the Apache-2.0 D3 release, pinned with source hashes in
[model_pins.json](model_pins.json). The reviewed formatter is downloaded into
ignored `vendor/` and checked before import. The Intern adapter imports the
release's reviewed, pinned prompt compiler. The Peekaboolean adapter imports
the reviewed, hash-checked upstream model/data helpers and preserves its scoring
convention. These sources are attributed in the respective adapter docstrings
and [report](REPORT.md).

- D3: [source/model](https://huggingface.co/vllm-sr/d3-lite), Apache-2.0.
- Intern-Decision: [source/model](https://huggingface.co/internlm/Intern-Decision-0.8B),
  Apache-2.0 and preserved Qwen notices.
- Peekaboolean: [code](https://github.com/bykof/peekaboolean), Apache-2.0;
  [weights](https://huggingface.co/bykof/peekaboolean-450m), CC-BY-NC-4.0.
  LiquidAI backbone terms also apply. Noncommercial weights are not a commercial
  deployment recommendation.
- The MM Grounding DINO decoder-head restoration follows the checkpoint layout
  documented in [Transformers issue 49529](https://github.com/huggingface/transformers/issues/49529)
  and [PR 49552](https://github.com/huggingface/transformers/pull/49552); it restores
  checkpoint values rather than introducing new learned parameters.
- Other model terms are linked from their authoritative model cards in the
  report. SAM3-LiteText's public card declares Apache-2.0; verify its underlying
  model terms before redistribution/deployment. The gated base SAM3 checkpoint
  was not downloaded and no access terms were accepted.

For Apache-2.0 material, the license is available from the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0) and
included as [Apache-2.0.txt](licenses/Apache-2.0.txt).
Retain upstream license files and notices if redistributing adapted source or
weights. The benchmark does not publish weights or imagery.
