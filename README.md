<h1 align="center">✨ PhyWorld-KGC ✨</h1>

<h3 align="center">PhyWorld-KGC: A Logic-Rule-Guided World Knowledge Model for Spatio-Temporal Knowledge Graph Completion</h3>

</div>

<div align="center">
  
[English](README.md) | [简体中文](./README_zh_CN.md)

</div>

<p align="center">
  <a href="https://github.com/your-username/PhyWorld-KGC"><img src="https://img.shields.io/badge/GitHub-PhyWorld--KGC-black" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/version-1.1.1-blue" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>



## 🌟 Introduction

Spatio-temporal knowledge graphs (STKGs) record not only semantic and structural information but also the **timestamps** and **geographic locations** of facts. Because missing data is inevitable during construction, spatio-temporal knowledge graph completion (STKGC)—predicting the missing entities of facts given spatio-temporal context—is a fundamental capability for downstream applications.

Existing methods exploit spatio-temporal signals too simplistically: they either compress each snapshot into a single vector or treat locations as static features, and thus struggle to cope with three core difficulties arising from intertwined spatio-temporal information:

1. **Heterogeneous observation forms**: coordinates, administrative-region text, and missing values coexist in the same graph;
2. **Joint evolution**: events depend on the joint variation of relations, time, and regions rather than on a single dimension;
3. **Geographic feasibility**: top-ranked candidates may imply impossible speeds or geographic mutual exclusion.

**PhyWorld-KGC** models STKGC as **rule-guided evolutionary simulation + constrained reasoning**, and addresses the above issues through three stages.

---

## 📊 Datasets

📥 Dataset download: [Google Drive](https://drive.google.com/drive/folders/1hKbNKnV2eWZK6CaH8FFQEoHLxq77Xlde?usp=sharing)

Six benchmarks, of which two `-Ext` are **extrapolation** settings (strictly split by time) and the rest are **interpolation** settings (random 80/10/10 split). code: stkgc

### File Format

The native layout is headerless tab-separated facts:

```
subject <TAB> relation <TAB> object <TAB> timestamp <TAB> location
```

`location` takes the form of `(lat,lon)`, administrative-region text (e.g., `United States`), or `∅`.


---

## 🙏 Acknowledgements

We thank the maintainers of the ICEWS, GDELT, and YAGO datasets for making the data publicly available for research.
