# <editable>
# 在此处添加算法描述
# </editable>
# conn：             数据库连接
# inputs：           输入数据集合，数据类型：list， 存储组件输入节点对应的数据，
#                    通过输入节点的key获取数据，例如配置的key为“input1”, 那么inputs$input1
#                    即为该节点对应的数据表
# params：           参数集合，数据类型：list， 存储，获取的规则与inputs一致。需要注意的是：
#                    params中参数的值都是字符类型的，需要在代码中进行数据类型转换，比如：
#                    as.integer(params$centers)
# outputs：          存储规则参见inputs
# reportFileName：   算法运行报告文件的存储路径
# 返回值(可选)：     如果函数用于训练模型，则必须返回模型对象

"""
columns
特征
进行操作的所有列。请选择数值型数据，如果勾选了非数值类型数据，则会自动过滤，下个组件可能无法获取所有列。
"""


def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	from sklearn.decomposition import PCA
	import numpy as np

	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['columns'] + ' from ' + inputs['data_in'])

	'''
	主成分分析
	'''
	data_in = data_in.select_dtypes(include=['number'])  # 筛选数值型数据
	n_samples, n_features = data_in.shape
	if not 1 <= int(params['n_components']) <= n_features:
	    raise ValueError("\n降维后的维数为%r,该值必须要在[1,%r]之间." % (int(params['n_components']), n_features))

	pca_model = PCA(n_components = int(params['n_components']))
	pca_model.fit(data_in)
	pca_model.explained_variance_ratio_

	# 执行降维
	data_out = pca_model.transform(data_in)
	columns = list(range(1, int(params['n_components'])+1))
	columns = ['comp_'+str(i) for i in columns]
	data_out = pd.DataFrame(data_out, columns = columns)
	data_out=np.around(data_out,decimals=4)

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>
