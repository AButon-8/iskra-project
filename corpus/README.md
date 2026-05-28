# Iskra Authorship Corpus (IAC)

## Overview

The **Iskra Authorship Corpus (IAC)** is a curated corpus of Russian revolutionary underground press texts from the late 19th and early 20th centuries, compiled for authorship attribution and stylometric analysis research.

The corpus was created as part of the MA thesis project *“Authorship Attribution of Russian Illegal Revolutionary Press: The Case of Iskra (1900–1905)”* at the National Research University Higher School of Economics (HSE University).

Version: 1.0
Year: 2026

---

## Author

**Anastasia Bogdanova**
MA Program “Digital Methods in the Humanities”
National Research University Higher School of Economics (HSE University)

---

## Corpus Composition

* Total number of texts: **154**

  * 145 texts with established authorship
  * 9 dubia texts
* Number of authors: **8**
* Total size: **627,850 words**
* Language: **Russian**

Texts are stored as individual `.txt` files in UTF-8 encoding and organized into folders by author.

---

## Chronological Coverage

The corpus includes texts dated between:

* **1888** (early Plekhanov)
* **1928** (late Trotsky)

The primary focus of the corpus is the period of the newspaper *Iskra* (1900–1905), with some chronological extensions required for corpus balancing and author representation.

---

## Text Processing

The original pre-reform Russian orthography was automatically normalized to modern Russian orthography using a custom preprocessing script.

Each file contains one text document.

---

## Corpus Design Principles

The corpus was compiled according to the following principles:

* reliable authorship attribution;
* genre comparability;
* chronological consistency (with justified exceptions);
* comparable text lengths across authors.

---

## Repository Structure

```text
corpus/
├── dubia/
├── krupskaya/
├── lenin/
├── martov/
├── plekhanov/
├── trotsky/
└── ...
```

Each folder corresponds to one attributed author.
Dubia texts are stored separately in the `dubia/` directory.

---

## Sources

For bibliographic information and source references for each text, see the `SourceName` column in:

```text
IAC_Metadata.csv
```

---

## Intended Use

The corpus may be used for:

* authorship attribution;
* stylometric analysis;
* corpus linguistics research;
* machine learning experiments on historical texts;
* feature extraction and classification tasks.

---

## Limitations

The corpus has several limitations that should be considered:

* author imbalance;
* incomplete coverage of historical materials;
* genre limitations;
* OCR and normalization errors;
* possible corpus contamination.

The corpus is intended primarily for research and educational purposes.

---

## License

The corpus is distributed under the
**CC BY-NC-SA 4.0**
(Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) license.

---

## Citation

If you use the corpus, please cite:

> Bogdanova A. *Iskra Authorship Corpus (IAC).* Version 1.0. 2026.
> Available at: https://github.com/AButon-8/iskra-project
