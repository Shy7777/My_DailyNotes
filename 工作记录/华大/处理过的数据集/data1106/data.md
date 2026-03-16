注意事项：注意裁剪patch的时候一定要注意看能不能完整裁剪，如果需要
筛选条件：细胞数量少于10、纯黑像素占比大于30%：
data1:总共129114，筛后78647，占比60.9%
	train：47188
	val：15729
	test：15730
data2:总共123930，筛后56291，占比45.4%
	train:33774
	val：11258
	test：11259
data3:总共28188，筛后8589，占比30.5%
	train:5153
	val：1717
	test：1719


不筛选细胞数量，只筛选黑色像素（包括边缘填充）：
	data1:总共129114，筛后99073，占比76.7%
	data2:总共123930，筛后95190，占比76.8%
	data3:总共28188，筛后22256，占比78.9%

### 总共图像数量
	data1:1594
	data2:1530
	data3:348

当前有问题
的
偏移：/home/share/huadjyin/home/wanghaoran/shy/code/data1106/show_me/batch_results_data1/15565d2eb98a11f0917d1070fda1cc3c.jpg
/home/share/huadjyin/home/wanghaoran/shy/code/data1106/show_me/batch_results_data1/341cdbcab98a11f0917d1070fda1cc3c.jpg
/home/share/huadjyin/home/wanghaoran/shy/code/data1106/show_me/batch_results_data1/2812ed6ab98a11f0917d1070fda1cc3c.jpg

### debug
测试多个通道打印时：
	正常打印输出为通道0开始，但是这个有12个通道所以要把所有通道加起来可视化

并行处理完成。生成的临时文件块数: 1539
总 Patch 数量: 78648
划分 data1 (共 78648 个)...
[train] 总样本数 47188。开始初始化磁盘文件...
[train] 初始化 raw_gene 临时落盘文件: /home/share/huadjyin/home/wanghaoran/shy/data/data1106_patch/data1/train/raw_gene_tmp.bin
[train] 正在构建文件索引...
[train] 共涉及 1529 个源文件。开始一次性遍历合并...
  -> 进度: 文件 50/1529 ...在这里的时候变得非常的慢，请在优化同时加快速度

