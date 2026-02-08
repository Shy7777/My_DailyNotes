细胞分割：
	现在有一张tif文件的H&E染色图像，大小至少2G，现在我要进行单个细胞级别的分割任务，现在已知每个细胞边界的25个像素点左右，可能会上下浮动，我想根据这些像素点进行细胞分割，由于经常报这个错：libdeflate_zlib_decompress returned LIBDEFLATE_INSUFFICIENT_SPACE，所以没办法直接处理整张tif文件，最好是分块读取，图中是csv的具体信息，一行是一个像素点，包括x和y在不同的列中，分别是2，3列，第4列代表细胞序列，根据细胞名称进行group操作，然后再读取他们的像素坐标，且像素坐标不为整数，请用亚像素轮廓平滑适配任意点数，要求以上，生成一个可执行的python文件

已知掩码后进行细胞分割：
	现在有tif的mask文件，现在要根据mask进行分割，首先要预处理mask文件实例化，要求生成的图片是png格式的，并且名称和CSV文件中的cell_id一致，注意生成的png文件是原图tif分割后的结果而不是mask.tif文件分割后的结果，并且处理libdeflate_zlib_decompress returned LIBDEFLATE_INSUFFICIENT_SPACE这个问题，来处理整张tif文件。要求以上，生成代码，并且要求路径使用全局变量，方便设置
用豆包生成代码，gpt5.2修改代码报错

已知掩码后进行细胞分割（zarr版本）：
	我现在需要单个分割后的细胞png图像但不要掩码图像而是原图像，现在我有这个zarr数据集以及H&E染色的tif图像，我该怎么做

生成json：cell.json
	生成一个json文件，数据格式为{
	"cell_id":123,
	"nucleus_xy":[],
	"bbox_xy":[],
	"img_path":"",
	"mask_path":"",
	"img_224_path":""
	}，其中cell_id为文件名称，具体存放在csv文件中的第一列；nucleus_xy数据存放在csv文件中的第二三列，具体csv文件路径存放在...；另一个csv文件路径为...，bbox_xy数据存放在另一个csv文件的第二、三列，是一个成对的数组形式，每张图片的cell_id对应这个csv文件的cell_id列，img_path路径是这个...目录下的文件，同样文件名对应的是cell_id；img_224_path路径是...目录下的文件，文件名同样对应的是cell_id。把所有路径设置为全局变量。

生成json：topk_genes.json
	生成一个json文件，数据格式为{"cell_id":"",
	"genes":["gene1","gene2","gene3"],
	"expr":[1,2,3]}，其中cell_id数据存放在barcodes.tsv文件中，每一行代表一个cell_id，genes数据可能有多个，放在features.tsv文件的第二列中，matrix.mtx这个文件存放所有信息，从第四行开始，第一列代表features.tsv文件的行号，把这行对应的基因名称放入genes中，注意matrix.mtx这个文件很多行代表一个细胞，这时候就要把他们的genes放到一起去，如["gene1","gene2","gene3"]，第二列代表细胞序列在barcodes.tsv中，1就代表第一行细胞，matrix.mtx文件的第三列代表该基因名的表达值，存入expr中，要和genes值相对应，生成代码

生成json文件：cell_all_info.json
	生成一个json文件，数据格式为{
	"image_path":"",
	"case_dir":"",
	"cell_id":"",
	"gene_idx":[],
	"gene_val":[],
	"cancer_type":"",
	"coordinate":""
	}，这是将所有的Xenium文件下的数据集做一个汇总，其中image_path数据存放在Xenium文件下的所有文件夹中的cells.json中，需要遍历，这里做硬遍历，我会给出每个数据集的路径，请你给我11个关于这个的路径设置，我把每个路径设置为全局变量，case_dir是每个数据集的一个大的目录，cell_id在topk_genes.jsonl中有对应的键和值，并且为了对每个数据集作区分，要在cell_id的值的前缀加上对应数据集文件的名称比如：preview-data-xenium-prime-gene-expression-aaaaljij-1，用-做拼接，对应的gene_idx和gene_val在topk_genes.jsonl对应的键是genes和expr，canncer_type在xenium.xlsx中的DISEASE列中有存，case_dir的值和这个xlsx文件的DATASET文件是对应的，最后的coordinate存放的值是cells.json文件的nucleus_xy的值，把他们联系起来的是cell_id，请生成代码，并且所有路径配置都设置为全局变量

{"image_path": "/home/share/huadjyin/home/wanghaoran/shy/data/Xenium/preview-data-xenium-prime-gene-expression", "case_dir": "/home/share/huadjyin/home/wanghaoran/shy/data/Xenium/preview-data-xenium-prime-gene-expression", "cell_id": "preview-data-xenium-prime-gene-expression-ajfnplhh-1", "gene_idx": ["AFF1", "AGAP3", "ALDH3A2", "ALG6", "ALKBH5", "AMER1", "AP1B1", "APH1B", "API5", "APOL6", "ARID1A", "ARL8B", "ATXN2L", "ATXN3", "BCL11B", "BEX2", "BTBD9", "BTLA", "CARF", "CASP8", "CBL", "CBX3", "CBX7", "CCT7", "CD3E", "CD3G", "CD4", "CD44", "CD84", "CD99", "CDC42", "CDV3", "CHD3", "CHRAC1", "CLEC16A", "CLK4", "CSNK1A1", "CXCR4", "CYLD", "DCAF1", "DDB1", "DDX39B", "DNAJC10", "DNM2", "DYRK1A", "EEF1G", "EFR3A", "ELK4", "ELMOD2", "ENOSF1", "ERCC5", "EWSR1", "FAIM", "FBXO45", "FLI1", "GIMAP4", "GLB1", "GLS", "GSR", "GTF2I", "HDAC1", "HDAC6", "HEXA", "HNRNPD", "HSDL1", "ICAM3", "IFIT3", "IFRD1", "IL6ST", "ITGAL", "ITK", "JAK1", "KHDRBS1", "KPNB1", "LAMA3", "LBH", "LDHA", "LENG8", "LEPROTL1", "LGALS8", "LINS1", "LONP1", "LRBA", "MAD1L1", "MAN1A1", "MAPRE2", "MARK2", "MAVS", "MDM2", "METTL2A", "MKRN2", "MMP9", "MPHOSPH9", "MSN", "MX1", "MYH9", "MYO1F", "NCOR1", "NCOR2", "NFKB2", "NOLC1", "NR1H3", "NSD2", "P4HA1", "PAK2", "PARP11", "PCNX3", "PDCD10", "PEA15", "PECAM1", "PHF23", "PITPNB", "PKM", "PLCG1", "POLR2E", "POLR2G", "PPP1R15B", "PPP1R9B", "PRDM1", "PRKAB1", "PRR13", "PSMA6", "PTGES3", "PTPN13", "PTPN6", "PURB", "R3HDM1", "RAB8A", "RAC1", "RAP1A", "RASSF5", "RBMS1", "RBMS3", "RGPD1", "RMND5A", "RSPRY1", "SART1", "SCFD2", "SDHA", "SDHB", "SFXN1", "SGSH", "SIRPB1", "SLC17A5", "SMARCA5", "SND1", "SOX2-OT", "SRSF1", "STAT5A", "STAU1", "STK39", "STXBP5", "SUV39H1", "SYNCRIP", "TAB1", "TCF3", "TELO2", "TERF2", "TNFRSF14", "TNFSF8", "TOM1L2", "TOMM40", "TP53BP2", "TRAF2", "TRIB2", "TRIM65", "TRMT1L", "UCP2", "UEVLD", "USP9Y", "UTRN", "VCAM1", "WAPL", "WDHD1", "WDR26", "XBP1", "XPO1", "ZAP70", "ZBTB1", "ZFP57", "ZMYND19", "ZNF106", "ZNF410"], "gene_val": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 2, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1], "cancer_type": "Lymph Node", "coordinate": [2353.490478515625, 1633.14013671875]}

全部数据集处理完成：总处理 5155360 行，总生成 5155360 条记录完成。

现在要生成一个types.npy文件，文件中装着patch对应的disease，它在xenium.xlsx文件中的DISEASE列中有存，Images.npy的第一个元素存的是patch_num，将patch_num作为该数组的下标，然后值就是xenium.xlsx中的对应DISEASE列，我会给你xenium.xlsx中一个列的字段值，然后可以匹配到对应的DISEASE，比如Images.npy有1000个patch_num，那现在生成的npy文件就有一个1000个元素的一维数组，里面的值就是对应的DISEASE，最后，便于赋值全局变量放到最上面

先生成整张的掩码再切分？切分之后连着image和mask一起过滤，image过滤哪个mask也过滤哪个

用python写一个要生成三个.npy文件，分别是raw_gene.npy，masks.npy，images.npy，其中，images.npy格式为[patch_num,256,256,3]要将整张的tiff文件分割成256✖️256大小的png格式patch，并对patch编号作为patch_num，3表示png格式的维度，这里的参考/home/share/huadjyin/home/wanghaoran/shy/data/code/patch.py的代码，它也是相应的操作，我会给你提供这张tiff对应的mask文件路径，但是masks.npy格式为[patch_num,256,256,6]其中这最后一个维度6，第一维存储的是细胞的id，你帮我对于每张patch给他们按顺序进行id的重新编码，就从0开始，随后中间2，3，4，5维度都设置为0，最后一个维度保存的是对应组织的代码编号，这里让我自己设置值就好，你弄成常数让我设置，然后就是raw_gene.npy文件，raw_gene.npy存储每个patch中的细胞和基因信息，每个patch一个维度，单个维度的格式为
{
	"has_omics":"True",
	"cell_data":[
		{
			"centroid":[],
			"is_valid":"True",
			"raw_expression":{
			"gene_name1":0,
			"gene_name2":0
			}
		}
	],
	"patch_gene_raw":{}
}，其中，has_omics永远是True，cell_data包含每一个细胞的数据，其中centroid为细胞的xy质心坐标，我会给一个cells.csv文件的路径，然后他在tiff中的质心坐标分别在第二列x_centroid和第三列y_centroid，此外由于需要配准，利用一个齐次坐标变换矩阵将当前文件的质心坐标转化为tiff图上的像素坐标，这里参考/home/share/huadjyin/home/wanghaoran/shy/code/seg/seg_cell_masks.py的代码，确认好细胞位置之后，这里我要的质心坐标是在每个patch中的质心坐标，所以还需要重新计算一下细胞在每个patch中的质心坐标，所以要明确哪些patch中有哪些细胞，is_valid永远是True，raw_expression是细胞的基因表达值，里面是每个细胞的对应基因名和基因表达值，对应的文件是matrix.mtx文件：前两行为注释，第三行表示统计数量总和，col1:基因序列号（对应features.tsv文件的行号，每行内容有col1基因编号和col2基因名称），col2:细胞序列号（对应barcodes.tsv文件的行号，每行内容为细胞的名称编号，这里的名称编号和cells.csv文件的第一列cell_id是相对应的），col3:基因表达值（对应细胞基因的基因表达值），所以我要的是基因名称和对应的基因表达值，最后patch_gene_raw存储的是这个patch中所有细胞的表达值之和。把以上所有需要设置路径的都设置为全局变量让我容易设置，并且中文注释



我会给你很多tiff图像和对应的mask文件以及h5ad文件，你根据命名相同的为一组，用python写一个要生成四个.npy文件，分别是raw_gene.npy，masks.npy，images.npy和types.npy，其中，images.npy格式为[patch_num,256,256,3]要将整张的tiff文件分割成256✖️256大小的png格式patch，并对patch编号作为patch_num，3表示png格式的维度，这里的参考/home/share/huadjyin/home/wanghaoran/shy/data/code/patch.py的代码，它也是相应的操作，我会给你提供这张tiff对应的mask文件路径，但是masks.npy格式为[patch_num,256,256,6]其中这最后一个维度6，第一维存储的是细胞的id，你帮我对于每张patch给他们按顺序进行id的重新编码，就从0开始，随后中间2，3，4，5维度都设置为0，最后一个维度保存的是对应组织的代码编号，这里让我自己设置值就好，你弄成常数让我设置，然后就是raw_gene.npy文件，raw_gene.npy存储每个patch中的细胞和基因信息，每个patch一个维度，单个维度的格式为
{
	"has_omics":"True",
	"cell_data":[
		{
			"centroid":[],
			"is_valid":"True",
			"raw_expression":{
			"gene_name1":0,
			"gene_name2":0
			}
		}
	],
	"patch_gene_raw":{}
}，其中，has_omics永远是True，cell_data包含每一个细胞的数据，其中centroid为细胞的xy质心坐标，我会给一个cells.csv文件的路径，然后他在tiff中的质心坐标分别在第二列x_centroid和第三列y_centroid，此外由于需要配准，利用一个齐次坐标变换矩阵将当前文件的质心坐标转化为tiff图上的像素坐标，这里参考/home/share/huadjyin/home/wanghaoran/shy/code/seg/seg_cell_masks.py的代码，确认好细胞位置之后，这里我要的质心坐标是在每个patch中的质心坐标，所以还需要重新计算一下细胞在每个patch中的质心坐标，所以要明确哪些patch中有哪些细胞，is_valid永远是True，raw_expression是细胞的基因表达值，里面是每个细胞的对应基因名和基因表达值，对应的文件是matrix.mtx文件：前两行为注释，第三行表示统计数量总和，col1:基因序列号（对应features.tsv文件的行号，每行内容有col1基因编号和col2基因名称），col2:细胞序列号（对应barcodes.tsv文件的行号，每行内容为细胞的名称编号，这里的名称编号和cells.csv文件的第一列cell_id是相对应的），col3:基因表达值（对应细胞基因的基因表达值），所以我要的是基因名称和对应的基因表达值，最后patch_gene_raw存储的是这个patch中所有细胞的表达值之和。把以上所有需要设置路径的都设置为全局变量让我容易设置，并且中文注释