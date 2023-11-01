## Induction Dataset

The Induction Mice are trained on the synthetically generated Induction dataset, which comes in two flavours: monogenic and polygenic. This dataset is designed to enable experiments that could potentially uncover induction heads in ViTs, akin to the induction heads found in language models.

## Monogenic Induction dataset

![Sample from each class](assets/images/monogenic_induction.png)

**Classes** 

0. **Vertical-Same**
1. **Vertical-Not Same**
2. **Horizontal-Same**
3. **Horizontal-Not same**

Here, the same/not-same indicates whether the structures present in the image are the same or not.

## Polygenic Induction dataset

![Sample from each class](assets/images/polygenic_induction.png)

**Classes**

| ID | Orientation | Pattern |
|----|-------------|---------|
| 0  | H           | AAAA    |
| 1  | H           | ABAB    |
| 2  | H           | ABBA    |
| 3  | H           | AABB    |
| 4  | H           | ABBB    |
| 5  | H           | AAAB    |
| 6  | V           | AAAA    |
| 7  | V           | ABAB    |
| 8  | V           | ABBA    |
| 9  | V           | AABB    |
| 10 | V           | ABBB    |
| 11 | V           | AAAB    |

The Orientation column indicates whether the structures are arranged horizontally (H) or vertically (V). The Pattern column denotes the sequence in which the structures are arranged.