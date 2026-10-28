<h1 align="center">✨ PhyWorld-KGC ✨</h1>

<h3 align="center">PhyWorld-KGC：面向时空知识图谱补全的逻辑规则引导世界知识模型</h3>

<p align="center">
  <a href="https://github.com/your-username/PhyWorld-KGC"><img src="https://img.shields.io/badge/GitHub-PhyWorld--KGC-black" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/version-1.1.1-blue" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> English README: [`README.md`](README.md)


## 🌟 简介

时空知识图谱（STKG）不仅记录语义与结构信息，还记录事实的**时间戳**与
**地理位置**。由于构建过程必然存在缺失，时空知识图谱补全（STKGC）——
在给定时空上下文下预测事实的缺失实体——是下游应用的基础能力。

现有方法对时空信号的利用过于简单：或将每个快照压缩成一个向量，或将位置
当作静态特征，因而难以应对交织时空信息的三个核心困难：

1. **观测形态异构**：坐标、行政区划文本、缺失值在同一张图中并存；
2. **联合演化**：事件取决于关系、时间、区域三者的联合变化，而非单一维度；
3. **地理可行性**：排名靠前的候选可能隐含不可能的速度或地理互斥。

**PhyWorld-KGC** 将 STKGC 建模为**规则引导的演化模拟 + 约束推理**，通过三个
阶段解决上述问题。

---

## 📊 数据集

六个基准，其中两个 `-Ext` 为**外推**设定（严格按时间切分），其余为**内插**
设定（随机 80/10/10 切分）。密码：stkgc

### 文件格式

原生布局为无表头的制表符分隔事实：

```
主语 <TAB> 关系 <TAB> 宾语 <TAB> 时间戳 <TAB> 位置
```

`位置` 取值为 `(lat,lon)`、行政区文本（如 `United States`）或 `∅`。


---

## 🙏 致谢

感谢 ICEWS、GDELT 与 YAGO 数据集的维护者将数据公开用于研究。
