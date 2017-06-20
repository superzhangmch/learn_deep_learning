### char rnn model use tensorflow ### 
对 http://karpathy.github.io/2015/05/21/rnn-effectiveness/ https://github.com/karpathy/char-rnn 的tensorflow 实现。
还参考了 https://github.com/sherjilozair/char-rnn-tensorflow

如果训练语料是 linux source, 试验发现在3层LSTM + 512维隐层下（learning_rate = 0.001, 且大约用到了230M的linux source），效果很好。可以说达到了和上面文章中说的效果（3层LSTM + 512维隐层也正是文章中的配置）。2层LSTM + 128维隐层下，效果不怎样。括号常有错误的配对，很难生成完整的c函数。
