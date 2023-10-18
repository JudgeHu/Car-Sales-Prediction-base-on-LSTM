# Car-Sales-Prediction-base-on-LSTM
SJTU International Marketing Team Project 2
# 1. 背景介绍 Background
我们项目会使用上汽大众的直播数据进行分析，并使用分析得出的数据设计改进营销方案。
经过对上汽大众战略的分析和实际数据的分析，我们初步得出了上汽大众接下来在华营销战略将向新能源车倾斜。
Our project will use SAIC Volkswagen's live data for analysis and use the data derived from the analysis to design improved marketing programs.
After analyzing SAIC Volkswagen's strategy and actual data, we have initially concluded that SAIC Volkswagen's next marketing strategy in China will be tilted towards new energy vehicles.

# 2. 建模目的 Modeling Purpose
我们获取了每月直播场次的数据，直播场次的分布不够合理，因此我们认为需要对每月的直播场次进行优化。本次开源的项目主要是为了解决我们在营销策略中时间维度的优化。
We obtained data on the number of monthly live broadcasts , the distribution of live broadcasts is not reasonable enough , so we think we need to optimize the number of monthly live broadcasts . The main purpose of this open source project is to solve the optimization of the time dimension in our marketing strategy.

# 3. 模型介绍 Model Introduction
我们将使用LSTM进行时间序列预测，使用预测的数据作为基础实现直播营销预算进行更加合理的调配。
We will use LSTM for time series prediction, and use the predicted data as a basis for realizing live marketing budgets for more rational deployment.

# 4. 开发环境介绍 Development Environment
如果您需要运行代码，请保证您的计算机部署了相关的环境。在该代码中，GPU不是必须的，但是GPU可以提高运算效率。
If you need to run the code, make sure you have the relevant environment deployed on your computer. GPUs are not required in this code, but GPUs can improve computing efficiency.
我将向您介绍我的软件与硬件环境，如下所示：
I will introduce you to my software and hardware environment as follows:


Software:

Python 3.9.0

Pytorch

Numpy

Pandas

Matplotlib

Sklearn

Hardware:
GPU: NVIDIA RTX 3080

CPU: Intel i5 12600K

RAM: 32G
