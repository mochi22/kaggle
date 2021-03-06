## Sartorius-Cell-InstanceSegmentation

### Introduction
kaggleに初参加したSartorius-Cell-Instance-Segmentationのコンペに参加しました。
本コンペについてのまとめです。
<br>

This is my first competition in kaggle 'Sartorius-Cell-InstanceSegmentation'.
I introduce it and my effort.

## Description
細胞のインスタンスセグメンテーションを行うことで簡単に細胞判別ができ、がんの治療などに貢献できるかもしれないコンペです。以下本コンペの説明です。
<br>
<br>
Neurological disorders, including neurodegenerative diseases such as Alzheimer's and brain tumors, are a leading cause of death and disability across the globe. However, it is hard to quantify how well these deadly disorders respond to treatment. One accepted method is to review neuronal cells via light microscopy, which is both accessible and non-invasive. Unfortunately, segmenting individual neuronal cells in microscopic images can be challenging and time-intensive. Accurate instance segmentation of these cells—with the help of computer vision—could lead to new and effective drug discoveries to treat the millions of people with these disorders.

Current solutions have limited accuracy for neuronal cells in particular. In internal studies to develop cell instance segmentation models, the neuroblastoma cell line SH-SY5Y consistently exhibits the lowest precision scores out of eight different cancer cell types tested. This could be because neuronal cells have a very unique, irregular and concave morphology associated with them, making them challenging to segment with commonly used mask heads.

Sartorius is a partner of the life science research and the biopharmaceutical industry. They empower scientists and engineers to simplify and accelerate progress in life science and bioprocessing, enabling the development of new and better therapies and more affordable medicine. They're a magnet and dynamic platform for pioneers and leading experts in the field. They bring creative minds together for a common goal: technological breakthroughs that lead to better health for more people.

In this competition, you’ll detect and delineate distinct objects of interest in biological images depicting neuronal cell types commonly used in the study of neurological disorders. More specifically, you'll use phase contrast microscopy images to train and test your model for instance segmentation of neuronal cells. Successful models will do this with a high level of accuracy.

If successful, you'll help further research in neurobiology thanks to the collection of robust quantitative data. Researchers may be able to use this to more easily measure the effects of disease and treatment conditions on neuronal cells. As a result, new drugs could be discovered to treat the millions of people with these leading causes of death and disability.

## Evaluation
本コンペの評価はIoU閾値における真陽性の割合を平均化しRLE(ランレングス符号化)にしたものを提出する。
<br>
<br>
This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

<img src="https://latex.codecogs.com/svg.image?IoU(A,B)=\frac{(A\cap&space;B)}{(A\cup&space;B)}" />

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value , a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

<img src="https://latex.codecogs.com/svg.image?\frac{TP}{TP&plus;FP&plus;FN}" title="\frac{TP}{TP+FP+FN}" />

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

<img src="https://latex.codecogs.com/svg.image?\frac{1}{thresholds}\sum&space;\frac{TP}{TP&plus;FP&plus;FN}" title="\frac{1}{thresholds}\sum \frac{TP}{TP+FP+FN}" />

Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

Submission File
In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted nucleus segmentation for the given ImageId. 

## myefforts
本コンペではDetectron2の転移学習させたmodelのスコアが非常に良かったためDetectron2をベースモデルとした。具体的にはU-NetやEfficientNetを使用したところスコアが0.25どまりであったが、Detectron2では0.3程度までスコアが到達した。
単体で0.3程度のスコアになったモデルを使用し、NMS, soft-NMS, NMW, WBFを単体モデル、複数モデルなどの全パターンに適応したところ複数モデルをNMWした場合が最もよく判別できていた。これは細胞を判別するときに、細胞の一部を複数のbboxで判別していたため、細胞全体で判断できたため最もスコアが良くなったと考えた。
最終的に

* できたこと

始めはよくわからなかったRLEなどもコードを一つずつ根気よく実行することで内容が分かっていった。
Detectron2は初めて触れたが、Detectron2のサイトにて調査しハイパラやclassなどを調節できた。

* できなかったこと

複数のmodelを使用することで相関の低いmodelができ、アンサンブルによってスコアが上がることが多い。しかし本コンペにおいて時間がなかったため、インスタンスセグメンテーションのアンサンブルの仕方がよくわからずDetectron2を使用したmodelしか使用できなかった。
半教師学習用のデータがあり使用してみたがスコアが0.1程度で、目視したところ検出できていなかった。つまり半教師学習の理解が足りなかった。

## reflection
結果としてスコアが0.311で330/1559位となった。銅メダルが0.320、銀メダルが0.332、金メダルが0.347以上と力不足であった。メダルを取ることがすべてではないがとても悔しい結果となった。
スコア的に、おそらく一つ以上のアイデアが必要なのが銅メダル、二つ以上のアイデアが必要なのが銀メダルだと思われる。CodeやDiscussionで話されているもの以上のアイデアが思い浮かばなかったので他の参加者のコードをみて血肉としたい。




