
# ChildAid AI — Intelligent Support System for Child Welfare Schemes

ChildAid AI is an early-stage project aimed at developing an intelligent system to support welfare eligibility and resource access for children in need, based on government and NGO schemes across India.

This repository contains the **initial prototype dataset** of child-centric welfare schemes in structured `.txt` format, to be used for training or querying via a language model (LLM). 

## Project Objectives

- Organize verified welfare schemes for children (0–18 years) into machine-readable text files.
- Tag schemes using relevant keywords for search/retrieval.
- Enable LLM-based retrieval and matching of user queries to available benefits.
- Eventually build a chatbot or search interface for NGOs, social workers, or government officers.

## Contents

- `data/`: Individual `.txt` files for each child welfare scheme, tagged and structured.
- `docs/scheme_format.md`: Format guide for writing and tagging schemes.
- `model/`: Placeholder for integration with LLM-based query engine (e.g., sentence-transformers + Flask app).

## Status

-  Base scheme data written and tagged manually.
-  Limited coverage (8–10 schemes for now).
-  Future plans include improving scheme parsing, expanding dataset, and frontend UI.

## Use Cases

- Matching children (orphans, disabled, rescued, etc.) to suitable support schemes.
- Helping case workers and field volunteers retrieve benefits faster.
- Building a national-level assistant for child welfare awareness and access.

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material
  
## Acknowledgements

- Ministry of Women and Child Development (MoWCD)
- National Commission for Protection of Child Rights (NCPCR)
- CHILDLINE India Foundation (1098)
- Social Justice and Empowerment Portal