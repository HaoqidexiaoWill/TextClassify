# TextClassification

| model              | MRR  | MAP   | Precision1 |
|--------------------|------|-------|------------|
| BertCAFE           | 0.56 | 0.52  | 0.37       |
| BertDPCNN(Conv2D)  | 0.56 | 0.530 | 0.37       |
| BertDPCNN(conv1D)  | 0.57 | 0.537 | 0.38       |
| BertLSTM           | 0.58 | 0.54  | 0.40       |
| BertHAN            | 0.581 | 0.545  | 0.384       |
| BertRCNN           | 0.59 | 0.55  | 0.41       |
|3输入BERT+CNN+cat    |0.63|0.58|0.46|
|3输入BERT+两种交互+CNN+cat|0.627 |0.578| 0.457|
|3输入BERT+TripleAtt+CNN+cat|0.6326 |0.5837| 0.4639|
|3输入BERT+TripleAtt+RNN+CNN+cat|0.6357|0.5862|0.4684|
|3输入BERT+TripleAtt+联合Att+meanmaxPooling|0.6348|0.5854|0.4639|
|3输入BERT+TripleAtt+Transformer+meanmaxPooling|0.6410| 0.5941|0.4714|
3输入BERT+TripleAtt+CNN+meanmaxPooling|0.6448|-|-|
