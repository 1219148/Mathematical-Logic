1.在nesylink中新建了shared、perception和agents文件夹，具体作用如下：

shared文件夹中定义了一些智能体决策必要的信息，感知模块应该提取这些信息

perception文件夹是感知模块，从像素中提取到上述信息，提供函数给智能体模块调用

agent文件夹是智能体模块，在内部调用感知模块对应函数获得相应信息，并使用信息完成决策

2.新建了lean文件夹，内有三个主要的lean文件,如下：

Environment.lean：类型、单步转移等等

Strategy.lean：多步执行、BFS、策略性质证明

TaskInitStates.lean：5 个关卡的初态编码