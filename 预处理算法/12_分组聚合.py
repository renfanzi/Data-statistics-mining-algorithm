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
features
特征
进行操作的所有列。当聚合方式为count外，请选择数值型数据。勾选的列不包含键

label
分组主键
分组的主键。按照该列的值将数据分组
"""



def execute(conn, inputs, params, outputs, reportFileName):
    #<editable>
	'''
	载入模块
	'''
	import pandas as pd
	import db_utils
	'''
	选择目标数据
	'''
	data_in = db_utils.query(conn, 'select '+ params['features'] + ',' + params['label'] + ' from ' + inputs['data_in'])

	'''
	分组聚合
	'''
	b = params['label'].split(',')
	if params['method'] == 'count':
	    data_out = data_in.groupby(b).count().reset_index()
	elif params['method'] == 'max':
	    data_out = data_in.groupby(b).max().reset_index()
	elif params['method'] == 'mean':
	    data_out = data_in.groupby(b).mean().reset_index()
	elif params['method'] == 'median':
	    data_out = data_in.groupby(b).median().reset_index()
	elif params['method'] == 'size':
	    data_out = data_in.groupby(b).size().reset_index()
	elif params['method'] == 'min':
	    data_out = data_in.groupby(b).min().reset_index()
	elif params['method'] == 'std':
	    data_out = data_in.groupby(b).std().reset_index()
	else :
	    data_out = data_in.groupby(b).sum().reset_index()

	'''
	将结果写出
	'''
	db_utils.dbWriteTable(conn, outputs['data_out'], data_out)
    #</editable>
