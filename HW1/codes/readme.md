# 代码补充说明
> 张天祺 2021010719 计12

在`codes/loss.py`文件中修改`FocalLoss`的初始化函数，添加对$\alpha$参数的初始化
在`codes/solve_net.py`文件中修改了一些函数的返回，从而能够更方便收集到loss和acc的数据
新增一个文件`run_mlp_pipeline.py`，为了进行批量测试。